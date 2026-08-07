// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for null propagation, constant folding, nullability widening, and options serde.

use super::*;
use crate::dtype::Nullability;
use crate::dtype::PType;

/// An `i32` element that is [dense-safe] iff `DENSE`, and otherwise the plain `i32` element in
/// every respect. Dense-safety is what decides the null-handling path, so a pair of these is
/// how one kernel gets run under both.
///
/// [dense-safe]: InputElement::DENSE_SAFE
struct MaybeDenseI32<const DENSE: bool>;

impl<const DENSE: bool> InputElement for MaybeDenseI32<DENSE> {
    type Column = <i32 as InputElement>::Column;
    type Varying<'a> = <i32 as InputElement>::Varying<'a>;
    type Elem<'a> = i32;

    const DENSE_SAFE: bool = DENSE;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i32 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        <i32 as InputElement>::decode(array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> i32 {
        <i32 as InputElement>::get(column, index)
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        <i32 as InputElement>::varying(column)
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        <i32 as InputElement>::varying_len(column)
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i32
    where
        Self: 'a,
    {
        <i32 as InputElement>::get_varying(column, index)
    }
}

/// Wrapping addition over two [`MaybeDenseI32`] columns.
#[derive(Clone)]
struct Add<const DENSE: bool>;

impl<const DENSE: bool> RowFn for Add<DENSE> {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        if DENSE {
            static ID: CachedId = CachedId::new("vortex.test.add.dense");
            *ID
        } else {
            static ID: CachedId = CachedId::new("vortex.test.add.filter");
            *ID
        }
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<
            (MaybeDenseI32<DENSE>, MaybeDenseI32<DENSE>),
            ElementSink<i32>,
            _,
            _,
        >(
            |_| (),
            |&(), (lhs, rhs), output| *output = lhs.wrapping_add(rhs),
        )
    }
}

/// Adds `lhs` to `rhs` under both null-handling paths and asserts each result equals
/// `expected`, which is what every case below does.
///
/// Forcing a _strategy_ within the filter contract is a separate axis, covered in
/// [`null_strategies`](super::null_strategies).
fn assert_add(lhs: ArrayRef, rhs: ArrayRef, expected: ArrayRef) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    let dense = apply(Add::<true>, [lhs.clone(), rhs.clone()], &mut ctx)?;
    let filtered = apply(Add::<false>, [lhs, rhs], &mut ctx)?;

    let args = [
        DType::Primitive(PType::I32, Nullability::NonNullable),
        DType::Primitive(PType::I32, Nullability::NonNullable),
    ];
    assert_eq!(policy(&Add::<true>, &args), RowPolicy::Dense);
    assert_eq!(
        policy(&Add::<false>, &args),
        RowPolicy::ValidOnly {
            filtered_decode_cost: 0
        }
    );
    assert_arrays_eq!(dense, expected, &mut ctx);
    assert_arrays_eq!(filtered, expected, &mut ctx);
    Ok(())
}

#[test]
fn no_nulls() -> VortexResult<()> {
    assert_add(
        PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
        PrimitiveArray::from_iter([10i32, 20, 30]).into_array(),
        PrimitiveArray::from_iter([11i32, 22, 33]).into_array(),
    )
}

#[test]
fn nulls_propagate() -> VortexResult<()> {
    assert_add(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), None]).into_array(),
        PrimitiveArray::from_option_iter([Some(10i32), Some(20), None, None]).into_array(),
        PrimitiveArray::from_option_iter([Some(11i32), None, None, None]).into_array(),
    )
}

/// Strictness: a null constant makes the whole output null without the kernel running at all.
#[test]
fn null_constant_short_circuits() -> VortexResult<()> {
    let null = Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable));

    assert_add(
        PrimitiveArray::from_iter([1i32, 2, 3]).into_array(),
        ConstantArray::new(null, 3).into_array(),
        PrimitiveArray::from_option_iter([Option::<i32>::None, None, None]).into_array(),
    )
}

/// All-constant inputs evaluate one row and broadcast it.
#[test]
fn all_constants_broadcast() -> VortexResult<()> {
    assert_add(
        ConstantArray::new(Scalar::from(2i32), 4).into_array(),
        ConstantArray::new(Scalar::from(40i32), 4).into_array(),
        PrimitiveArray::from_iter([42i32, 42, 42, 42]).into_array(),
    )
}

#[test]
fn mixed_constant_and_column() -> VortexResult<()> {
    assert_add(
        PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
        ConstantArray::new(Scalar::from(10i32), 3).into_array(),
        PrimitiveArray::from_option_iter([Some(11i32), None, Some(13)]).into_array(),
    )
}

/// An empty batch is neither all-valid nor all-null, and a zero-length non-nullable execution
/// keeps its non-nullable dtype.
#[test]
fn empty_input_keeps_dtype() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let empty = || PrimitiveArray::from_iter(Vec::<i32>::new()).into_array();

    let result = apply(Add::<false>, [empty(), empty()], &mut ctx)?;

    assert_eq!(result.len(), 0);
    assert!(!result.dtype().is_nullable());
    Ok(())
}

/// The output element dtype is non-nullable, and the lifting widens it iff an input is
/// nullable, which is what makes strictness's dtype contract hold by construction.
#[test]
fn return_dtype_unions_nullability() -> VortexResult<()> {
    let non_nullable = DType::Primitive(PType::I32, Nullability::NonNullable);
    let nullable = non_nullable.as_nullable();

    assert_eq!(
        ScalarFnVTable::return_dtype(
            &Add::<true>,
            &EmptyOptions,
            &[non_nullable.clone(), non_nullable.clone()]
        )?,
        non_nullable
    );
    assert_eq!(
        ScalarFnVTable::return_dtype(
            &Add::<true>,
            &EmptyOptions,
            &[non_nullable, nullable.clone()]
        )?,
        nullable
    );
    Ok(())
}

#[test]
fn a_row_fn_is_strict() {
    assert!(ScalarFnVTable::is_strict(&Add::<true>, &EmptyOptions));
}

/// Output sinks build an all-valid column, so the output validity is exactly the child
/// conjunction and the planner never has to execute the function to learn which rows are null.
#[test]
fn validity_is_the_child_conjunction() -> VortexResult<()> {
    let expr = Add::<true>.new_expr(EmptyOptions, [root(), root()]);

    assert!(ScalarFnVTable::validity(&Add::<true>, &EmptyOptions, &expr)?.is_some());
    Ok(())
}

/// A row function is not serializable until the function opts into a wire representation.
#[test]
fn options_are_not_serializable_by_default() -> VortexResult<()> {
    assert_eq!(
        ScalarFnVTable::serialize(&Add::<true>, &EmptyOptions)?,
        None
    );
    assert!(ScalarFnVTable::deserialize(&Add::<true>, &[], &array_session()).is_err());
    Ok(())
}
