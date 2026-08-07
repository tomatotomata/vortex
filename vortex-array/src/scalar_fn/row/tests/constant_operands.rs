// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests that constant operands are decoded once and broadcast across the batch.

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_buffer::Buffer;

use super::*;

/// Total rows handed to [`CountedI64::decode`] across one execution. Sound as a global because
/// each test binary runs one test per process.
static DECODED_ROWS: AtomicUsize = AtomicUsize::new(0);

/// Stands in for an element whose decode is expensive per row, recording how wide a column each
/// decode was actually given.
struct CountedI64;

impl InputElement for CountedI64 {
    type Column = Buffer<i64>;
    type Varying<'a> = <i64 as InputElement>::Varying<'a>;
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        DECODED_ROWS.fetch_add(array.len(), Ordering::Relaxed);
        <i64 as InputElement>::decode(array, ctx)
    }

    fn get(column: &Self::Column, index: usize) -> i64 {
        <i64 as InputElement>::get(column, index)
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        <i64 as InputElement>::varying(column)
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        <i64 as InputElement>::varying_len(column)
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
    where
        Self: 'a,
    {
        <i64 as InputElement>::get_varying(column, index)
    }
}

#[derive(Clone)]
struct AddCounted;

impl RowFn for AddCounted {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.add_counted");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(CountedI64, CountedI64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| *output = lhs + rhs,
        )
    }
}

/// An element whose decode drops the last row, standing in for a buggy element implementation.
struct ShortDecodeI64;

impl InputElement for ShortDecodeI64 {
    type Column = Buffer<i64>;
    type Varying<'a> = <i64 as InputElement>::Varying<'a>;
    type Elem<'a> = i64;

    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        <i64 as InputElement>::validate(dtype)
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let column = <i64 as InputElement>::decode(array, ctx)?;
        Ok(column.slice(0..column.len().saturating_sub(1)))
    }

    fn get(column: &Self::Column, index: usize) -> i64 {
        <i64 as InputElement>::get(column, index)
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        <i64 as InputElement>::varying(column)
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        <i64 as InputElement>::varying_len(column)
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> i64
    where
        Self: 'a,
    {
        <i64 as InputElement>::get_varying(column, index)
    }
}

/// Pairs a short-decoding argument with an ordinary one, so a batch-constant second operand takes
/// the mixed constant-and-varying read path rather than the all-varying one.
#[derive(Clone)]
struct AddShort;

impl RowFn for AddShort {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.add_short");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(ShortDecodeI64, i64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| *output = lhs + rhs,
        )
    }
}

/// A constant operand makes the whole tuple decline the all-varying read path, so the row loop
/// indexes each [`ArgColumn`](crate::scalar_fn::ArgColumn) directly. The decoded length still has to
/// be checked there, or a short column reaches an out-of-bounds row read.
#[test]
fn a_short_decode_beside_a_constant_operand_is_rejected() {
    let mut ctx = array_session().create_execution_ctx();
    let column = PrimitiveArray::from_iter(0..64i64).into_array();
    let constant = ConstantArray::new(Scalar::from(10i64), 64).into_array();

    let error = apply(AddShort, [column, constant], &mut ctx).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("does not address exactly 64 rows"),
        "{error}"
    );
}

#[test]
fn a_constant_operand_is_decoded_once() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let column = PrimitiveArray::from_iter(0..64i64).into_array();
    let constant = ConstantArray::new(Scalar::from(10i64), 64).into_array();

    let result = apply(AddCounted, [column, constant], &mut ctx)?;

    // 64 rows for the real column, plus exactly one for the constant.
    assert_eq!(DECODED_ROWS.load(Ordering::Relaxed), 65);
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter((0..64i64).map(|value| value + 10)),
        &mut ctx
    );
    Ok(())
}
