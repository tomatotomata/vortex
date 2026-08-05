// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution for primitive [`Interleave`] values.

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;

use super::super::Interleave;
use super::super::InterleaveArrayExt;
use super::validate_selectors;
use crate::array::Array;
use crate::array::ArrayView;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::NativePType;
use crate::executor::ExecutionCtx;
use crate::executor::ExecutionResult;
use crate::match_each_native_ptype;
use crate::match_each_unsigned_integer_ptype;
use crate::require_child;

pub(super) fn execute(
    array: Array<Interleave>,
    _ctx: &mut ExecutionCtx,
) -> VortexResult<ExecutionResult> {
    let num_values = array.num_values();
    let mut array = array;
    array = require_child!(array, array.array_indices(), 0 => Primitive);
    array = require_child!(array, array.row_indices(), 1 => Primitive);
    for i in 0..num_values {
        array = require_child!(array, array.value(i), i + 2 => Primitive);
    }

    let validity = array.as_ref().validity()?;
    let output = match_each_native_ptype!(array.value(0).as_::<Primitive>().ptype(), |T| {
        let values = gather_values::<T>(&array)?;
        VortexResult::Ok(PrimitiveArray::new(values, validity))
    })?;

    Ok(ExecutionResult::done(output))
}

fn gather_values<T: NativePType>(array: &Array<Interleave>) -> VortexResult<Buffer<T>> {
    let buffers = (0..array.num_values())
        .map(|i| array.value(i).as_::<Primitive>().to_buffer::<T>())
        .collect::<Vec<_>>();
    let branches = array.array_indices().as_::<Primitive>();
    let rows = array.row_indices().as_::<Primitive>();

    match_each_unsigned_integer_ptype!(branches.ptype(), |A| {
        gather_rows::<T, A>(&buffers, branches.as_slice::<A>(), rows)
    })
}

fn gather_rows<T, A>(
    values: &[Buffer<T>],
    branches: &[A],
    rows: ArrayView<'_, Primitive>,
) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
{
    match_each_unsigned_integer_ptype!(rows.ptype(), |R| {
        gather(values, branches, rows.as_slice::<R>())
    })
}

fn gather<T, A, R>(values: &[Buffer<T>], branches: &[A], rows: &[R]) -> VortexResult<Buffer<T>>
where
    T: NativePType,
    A: AsPrimitive<usize>,
    R: AsPrimitive<usize>,
{
    let len = validate_selectors(values.len(), |branch| values[branch].len(), branches, rows)?;
    let mut output = BufferMut::with_capacity(len);
    for i in 0..len {
        output.push(values[branches[i].as_()][rows[i].as_()]);
    }
    Ok(output.freeze())
}
