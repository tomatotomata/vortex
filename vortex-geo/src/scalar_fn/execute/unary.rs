// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Unary operand dispatch for native geometry kernels.

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Execution;
use super::Operand;

/// Dispatch a unary strict geometry kernel over a constant or column.
///
/// A null constant or all-null column short-circuits to an all-null constant output. Otherwise,
/// `kernel` receives the operand shape and its valid-row mask. The kernel remains responsible for
/// interpreting the native input and constructing its Vortex output.
pub(crate) fn dispatch_unary<K>(
    array: &ArrayRef,
    output_dtype: DType,
    kernel: K,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    K: FnOnce(Execution<1>, &mut ExecutionCtx) -> VortexResult<ArrayRef>,
{
    let len = array.len();
    if let Some(constant) = array.as_opt::<Constant>() {
        if constant.scalar().is_null() {
            return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
        }
        return kernel(
            Execution {
                operands: [Operand::Constant(constant.scalar().clone())],
                valid: Mask::new_true(len),
                len,
                nullability: output_dtype.nullability(),
            },
            ctx,
        );
    }

    let valid = array.validity()?.execute_mask(len, ctx)?;
    if len != 0 && valid.all_false() {
        return Ok(ConstantArray::new(Scalar::null(output_dtype), len).into_array());
    }
    kernel(
        Execution {
            operands: [Operand::Column(array.clone())],
            valid,
            len,
            nullability: output_dtype.nullability(),
        },
        ctx,
    )
}
