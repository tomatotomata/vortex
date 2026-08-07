// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for row functions that write into a batch-wide output sink.

use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;

use super::*;
use crate::arrays::FixedSizeListArray;
use crate::dtype::NativePType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

/// Builds a `FixedSizeList<T, W>` column, presenting each row as the `&mut [T]` slice to fill.
///
/// Its element dtype comes from the input rather than from `T` alone, so it exercises
/// [`OutputSink::sink_dtype`] actually reading `args`.
struct SpreadSink<T, const W: usize> {
    dtype: DType,
    rows: usize,
    elements: BufferMut<T>,
}

impl<T: NativePType, const W: usize> OutputSink for SpreadSink<T, W> {
    type Rows<'a> = (&'a mut [T], usize);
    type Row<'a> = &'a mut [T];

    fn sink_dtype(args: &[DType]) -> VortexResult<DType> {
        let element = args
            .first()
            .ok_or_else(|| vortex_err!("a spread sink takes its element dtype from its input"))?;
        <T as InputElement>::validate(element)?;
        Ok(DType::FixedSizeList(
            Arc::new(element.as_nonnullable()),
            u32::try_from(W).vortex_expect("test width fits in u32"),
            Nullability::NonNullable,
        ))
    }

    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            dtype: dtype.clone(),
            rows,
            elements: BufferMut::zeroed(rows * W),
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        (self.elements.as_mut_slice(), self.rows)
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.1 == row_count && row_count.checked_mul(W) == Some(rows.0.len())
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> &'a mut [T] {
        &mut rows.0[index * W..][..W]
    }

    fn finish(self, _error: DeferredError) -> VortexResult<ArrayRef> {
        vortex_ensure_eq!(
            self.dtype,
            Self::sink_dtype(&[DType::Primitive(T::PTYPE, self.dtype.nullability())])?,
            "the sink must build the dtype it named",
        );
        Ok(FixedSizeListArray::try_new(
            PrimitiveArray::new(self.elements.freeze(), Validity::NonNullable).into_array(),
            u32::try_from(W).vortex_expect("test width fits in u32"),
            Validity::NonNullable,
            self.rows,
        )?
        .into_array())
    }
}

/// A sink that reports a data-dependent error only after every row has been written.
struct NonNegativeSink {
    values: BufferMut<i64>,
}

struct NonNegativeRow<'a> {
    value: &'a mut i64,
}

impl NonNegativeRow<'_> {
    fn write(self, value: i64) -> bool {
        *self.value = value;
        value < 0
    }
}

impl OutputSink for NonNegativeSink {
    const ERRORS_ARE_DEFERRED: bool = true;

    type Rows<'a> = &'a mut [i64];
    type Row<'a> = NonNegativeRow<'a>;

    fn sink_dtype(args: &[DType]) -> VortexResult<DType> {
        let dtype = args
            .first()
            .ok_or_else(|| vortex_err!("a non-negative sink requires one input"))?;
        <i64 as InputElement>::validate(dtype)?;
        Ok(DType::Primitive(PType::I64, Nullability::NonNullable))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            values: BufferMut::zeroed(rows),
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.values.as_mut_slice()
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        NonNegativeRow {
            value: &mut rows[index],
        }
    }

    fn finish(self, error: DeferredError) -> VortexResult<ArrayRef> {
        if error.occurred() {
            vortex_bail!("negative output");
        }
        Ok(PrimitiveArray::new(self.values.freeze(), Validity::NonNullable).into_array())
    }
}

/// Broadcasts each input value across a fixed-size list row: `spread(x) == [x, x, x]`.
#[derive(Clone)]
struct Spread;

impl RowFn for Spread {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.spread");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), SpreadSink<i64, 3>, _, _>(
            |_| (),
            |&(), (x,), out| out.fill(x),
        )
    }
}

/// The same, but refusing negative inputs, so its row closure returns `VortexResult<()>`.
#[derive(Clone)]
struct SpreadNonNegative;

impl RowFn for SpreadNonNegative {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.spread_non_negative");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), SpreadSink<i64, 3>, _, _>(
            |_| (),
            |&(), (x,), out| {
                if x < 0 {
                    vortex_bail!("negative input {x}");
                }
                out.fill(x);
                Ok(())
            },
        )
    }
}

/// Writes infallibly and lets its output sink report the error after the loop.
#[derive(Clone)]
struct DeferredNonNegative;

impl RowFn for DeferredNonNegative {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.deferred_non_negative");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), NonNegativeSink, _, _>(
            |_| (),
            |&(), (value,), output| output.write(value),
        )
    }
}

/// `SpreadSink`'s three-element rows, built from `values`.
fn spread_rows(values: impl IntoIterator<Item = Option<i64>>) -> VortexResult<ArrayRef> {
    let values = values.into_iter().collect::<Vec<_>>();
    let rows = values.len();
    let flat = values
        .iter()
        .flat_map(|value| [value.unwrap_or(0); 3])
        .collect::<Vec<_>>();
    let validity = if values.iter().all(Option::is_some) {
        Validity::NonNullable
    } else {
        Validity::from_iter(values.iter().map(Option::is_some))
    };

    Ok(FixedSizeListArray::try_new(
        PrimitiveArray::new(flat, Validity::NonNullable).into_array(),
        3,
        validity,
        rows,
    )?
    .into_array())
}

/// The output dtype is the sink's, with its width, and every row holds the written slice.
#[test]
fn writes_one_row_at_a_time() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![7i64, -2, 0].into_array();

    let result = apply(Spread, [input], &mut ctx)?;

    assert_arrays_eq!(result, spread_rows([Some(7), Some(-2), Some(0)])?, &mut ctx);
    Ok(())
}

/// A null input row is written densely and masked away afterwards, exactly as on the value path.
#[test]
fn nulls_are_masked_after_the_sink() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::from_option_iter([Some(7i64), None, Some(4)]).into_array();

    let result = apply(Spread, [input], &mut ctx)?;

    assert!(result.dtype().is_nullable());
    assert_arrays_eq!(result, spread_rows([Some(7), None, Some(4)])?, &mut ctx);
    Ok(())
}

/// A sink whose closure cannot fail is dense, while one whose closure can fail is valid-only.
#[test]
fn null_handling_follows_from_declared_fallibility() {
    let args = [DType::Primitive(PType::I64, Nullability::NonNullable)];
    assert_eq!(policy(&Spread, &args), RowPolicy::Dense);
    assert!(!ScalarFnVTable::is_fallible(&Spread, &EmptyOptions));

    assert_eq!(
        policy(&SpreadNonNegative, &args),
        RowPolicy::ValidOnly {
            filtered_decode_cost: 0
        }
    );
    assert!(ScalarFnVTable::is_fallible(
        &SpreadNonNegative,
        &EmptyOptions
    ));
}

/// An error from a writing closure aborts the batch rather than being written into the sink.
#[test]
fn a_failing_row_propagates() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![1i64, -5, 3].into_array();

    let error = apply(SpreadNonNegative, [input], &mut ctx).unwrap_err();

    assert!(error.to_string().contains("negative input -5"), "{error}");
    Ok(())
}

/// A sink may accumulate a failure while its row closure remains infallible.
#[test]
fn a_sink_can_defer_its_error_until_finish() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![1i64, -5, 3].into_array();

    let error = apply(DeferredNonNegative, [input], &mut ctx).unwrap_err();

    assert!(error.to_string().contains("negative output"), "{error}");
    Ok(())
}

/// A deferred failure behind a null triggers a valid-row retry and is then discarded.
#[test]
fn a_deferred_error_behind_a_null_is_ignored() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::new(
        buffer![1i64, -5, 3],
        Validity::from_iter([true, false, true]),
    )
    .into_array();

    let result = apply(DeferredNonNegative, [input], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(1i64), None, Some(3)]),
        &mut ctx
    );
    Ok(())
}

/// Being fallible, `SpreadNonNegative` is filtered, so its closure never sees the value behind a
/// null row. A negative payload there must therefore not raise.
#[test]
fn a_failing_row_is_never_reached_behind_a_null() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input = PrimitiveArray::new(
        buffer![1i64, -5, 3],
        Validity::from_iter([true, false, true]),
    )
    .into_array();

    let result = apply(SpreadNonNegative, [input], &mut ctx)?;

    assert_arrays_eq!(result, spread_rows([Some(1), None, Some(3)])?, &mut ctx);
    Ok(())
}

/// The sink names its output dtype from the input, so a wrong input dtype is rejected at plan
/// time rather than producing a mis-typed column.
#[test]
fn the_sink_dtype_validates_its_input() {
    let dtype = DType::Primitive(PType::F64, Nullability::NonNullable);
    assert!(ScalarFnVTable::return_dtype(&Spread, &EmptyOptions, &[dtype]).is_err());
}

/// The width the sink declares is the width it builds, over the element dtype it read off the
/// input.
#[test]
fn the_return_dtype_is_the_sinks() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
    assert_eq!(
        ScalarFnVTable::return_dtype(&Spread, &EmptyOptions, std::slice::from_ref(&dtype))?,
        DType::FixedSizeList(Arc::new(dtype), 3, Nullability::NonNullable),
    );
    Ok(())
}
