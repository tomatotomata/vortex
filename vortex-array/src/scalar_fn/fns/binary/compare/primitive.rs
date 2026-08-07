// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native comparison of primitive arrays via bit-packing lane kernels.

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::compare::collect_bits;
use crate::scalar_fn::fns::binary::compare::collect_zip_bits;
use crate::scalar_fn::fns::binary::compare::compare_validity;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::validity::Validity;

/// Compare two primitive arrays of the same [`PType`].
///
/// Floats compare with Vortex's total ordering: `NaN` is the largest value, `-0.0 < +0.0`, and
/// equality is bitwise.
pub(super) fn compare_primitive(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PType::try_from(lhs.dtype())?;
    match_each_native_ptype!(ptype, |T| {
        compare_primitive_typed::<T>(lhs, rhs, op, nullability, ctx)
    })
}

fn compare_primitive_typed<T: NativePType>(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: CompareOperator,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let len = lhs.len();
    let lhs = PrimitiveOperand::<T>::try_new(lhs, ctx)?;
    let rhs = PrimitiveOperand::<T>::try_new(rhs, ctx)?;
    if lhs.len() != rhs.len() {
        vortex_bail!(
            "compare operator requires equal lengths, got {} and {}",
            lhs.len(),
            rhs.len()
        );
    }

    let validity = compare_validity(lhs.validity(), rhs.validity(), nullability)?;

    let bits = match (&lhs, &rhs) {
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slices(lhs, rhs, op),
        (
            PrimitiveOperand::Array { values: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => compare_slice_constant(lhs, *rhs, op),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Array { values: rhs, .. },
        ) => compare_slice_constant(rhs, *lhs, op.swap()),
        (
            PrimitiveOperand::Constant { value: lhs, .. },
            PrimitiveOperand::Constant { value: rhs, .. },
        ) => {
            // Unreachable through `execute_compare` (constant-constant is folded there), but
            // cheap to answer anyway.
            BitBuffer::full(apply_op(*lhs, *rhs, op), len)
        }
        (PrimitiveOperand::Null(_), _) | (_, PrimitiveOperand::Null(_)) => {
            return Ok(
                ConstantArray::new(Scalar::null(DType::Bool(Nullability::Nullable)), len)
                    .into_array(),
            );
        }
    };

    Ok(BoolArray::try_new(bits, validity)?.into_array())
}

#[inline(always)]
fn apply_op<T: NativePType>(lhs: T, rhs: T, op: CompareOperator) -> bool {
    match op {
        CompareOperator::Eq => lhs.is_eq(rhs),
        CompareOperator::NotEq => !lhs.is_eq(rhs),
        CompareOperator::Gt => lhs.is_gt(rhs),
        CompareOperator::Gte => lhs.is_ge(rhs),
        CompareOperator::Lt => lhs.is_lt(rhs),
        CompareOperator::Lte => lhs.is_le(rhs),
    }
}

fn compare_slices<T: NativePType>(lhs: &[T], rhs: &[T], op: CompareOperator) -> BitBuffer {
    // Dispatch the operator outside the lane loop so each instantiation vectorizes a single
    // branch-free predicate.
    match op {
        CompareOperator::Eq => collect_zip_bits(lhs, rhs, |a: T, b: T| a.is_eq(b)),
        CompareOperator::NotEq => collect_zip_bits(lhs, rhs, |a: T, b: T| !a.is_eq(b)),
        CompareOperator::Gt => collect_zip_bits(lhs, rhs, T::is_gt),
        CompareOperator::Gte => collect_zip_bits(lhs, rhs, T::is_ge),
        CompareOperator::Lt => collect_zip_bits(lhs, rhs, T::is_lt),
        CompareOperator::Lte => collect_zip_bits(lhs, rhs, T::is_le),
    }
}

fn compare_slice_constant<T: NativePType>(lhs: &[T], rhs: T, op: CompareOperator) -> BitBuffer {
    match op {
        CompareOperator::Eq => collect_bits(lhs, |a: T| a.is_eq(rhs)),
        CompareOperator::NotEq => collect_bits(lhs, |a: T| !a.is_eq(rhs)),
        CompareOperator::Gt => collect_bits(lhs, |a: T| a.is_gt(rhs)),
        CompareOperator::Gte => collect_bits(lhs, |a: T| a.is_ge(rhs)),
        CompareOperator::Lt => collect_bits(lhs, |a: T| a.is_lt(rhs)),
        CompareOperator::Lte => collect_bits(lhs, |a: T| a.is_le(rhs)),
    }
}

/// A primitive binary-operator operand: a materialized buffer, a non-null constant, or an
/// all-null constant.
///
/// Splitting the constant out of the buffer is what lets the lane kernels above hoist it into a
/// register instead of reading it back per lane.
enum PrimitiveOperand<T: NativePType> {
    /// A decoded column, one value per row.
    Array {
        values: Buffer<T>,
        validity: Validity,
    },

    /// The same non-null value in every row.
    Constant {
        value: T,
        len: usize,
        validity: Validity,
    },

    /// A null in every row, carrying only the row count.
    Null(usize),
}

impl<T: NativePType> PrimitiveOperand<T> {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            return Ok(
                match constant.scalar().as_primitive().try_typed_value::<T>()? {
                    Some(value) => Self::Constant {
                        value,
                        len: array.len(),
                        validity: if constant.scalar().dtype().is_nullable() {
                            Validity::AllValid
                        } else {
                            Validity::NonNullable
                        },
                    },
                    None => Self::Null(array.len()),
                },
            );
        }

        let array = array.clone().execute::<PrimitiveArray>(ctx)?;
        let validity = array.validity()?;
        let values = array.into_buffer::<T>();
        Ok(Self::Array { values, validity })
    }

    fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } => validity.clone(),
            Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}
