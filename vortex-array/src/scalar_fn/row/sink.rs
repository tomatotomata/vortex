// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The column builders a row function can write its output into.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::scalar_fn::DeferredError;
use crate::scalar_fn::OutputElement;

/// A column allocated once per batch that a row closure writes into, one row at a time.
///
/// Every [`RowFn`](crate::scalar_fn::RowFn) writes through one. [`ElementSink`] covers an ordinary
/// owned value per row. A custom sink covers output whose width is runtime data or whose rows append
/// into one batch-wide builder.
///
/// Two properties of the contract are worth stating, since both are load-bearing:
///
/// - **The row loop, not the closure, holds the sink.** [`row`](Self::row) is called by the framework
///   and its result passed in, so a writing closure stays [`Fn`] and captures nothing mutable.
///   Relaxing the row closure to `FnMut` instead was measured at 8 to 11%, because a captured `&mut`
///   inhibits vectorization of the loop.
/// - **[`sink_dtype`](Self::sink_dtype) sees the input dtypes**, unlike
///   [`OutputElement::element_dtype`](crate::scalar_fn::OutputElement::element_dtype), which takes
///   none. That is the whole reason a runtime-shaped output fits here: the width comes out of the
///   arguments.
///
/// Rows arrive in increasing index order. Ordinary execution visits `0..row_count` exactly once;
/// branch-and-skip may omit null rows when [`SUPPORTS_SKIPPED_ROWS`](Self::SUPPORTS_SKIPPED_ROWS)
/// is `true`.
pub trait OutputSink: 'static + Sized {
    /// Whether this sink accepts [`DeferredError`] from its row closure instead of requiring a
    /// per-row [`VortexResult`].
    ///
    /// The executor OR-reduces the row error words and passes the result to
    /// [`finish`](Self::finish). When the arguments are safe to read behind nulls, this lets the
    /// lifting optimistically run a dense loop. If `finish` reports the deferred error for a
    /// nullable batch, the lifting retries over only the valid rows: success means the error came
    /// exclusively from null rows, while another deferred error is real.
    ///
    /// A supporting sink must return an error from `finish` when its `error` argument occurred.
    const ERRORS_ARE_DEFERRED: bool = false;

    /// Whether this sink can finish a full-length output when some rows were never visited.
    ///
    /// A supporting sink must leave a legal arbitrary value at every skipped row. The lifting masks
    /// those rows before the result escapes, so that value is never observable.
    const SUPPORTS_SKIPPED_ROWS: bool = false;

    /// A loop-local view of all output rows.
    ///
    /// Borrowed once before execution so the sink's buffer descriptor and shape become loop
    /// invariants rather than being re-read through `&mut Self` for every row.
    type Rows<'a>
    where
        Self: 'a;

    /// The place a row closure writes one row through, borrowed from the sink.
    type Row<'a>
    where
        Self: 'a;

    /// The dtype of the column this sink builds, given the function's input dtypes.
    ///
    /// Must be non-nullable: nullability is derived from the inputs by the lifting, which
    /// widens the result and masks the null rows itself.
    fn sink_dtype(args: &[DType]) -> VortexResult<DType>;

    /// Allocate a sink for `rows` rows of `dtype`, which is this sink's own
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch.
    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;

    /// Borrow all output rows for the hot loop.
    fn rows(&mut self) -> Self::Rows<'_>;

    /// Whether every index in `0..row_count` is addressable through [`row`](Self::row).
    ///
    /// Called once before the hot loop. Besides validating the sink contract, this gives the
    /// optimizer the output bounds it needs to remove the bounds check hidden in each row accessor.
    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool;

    /// Hand out the place to write row `index`. Must be `O(1)`: it is called in the row loop.
    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a>;

    /// Finish into the built column, whose dtype **must** be this sink's
    /// [`sink_dtype`](Self::sink_dtype). Called once per batch with the OR of every row's deferred
    /// error bit.
    fn finish(self, error: DeferredError) -> VortexResult<ArrayRef>;
}

/// The standard output sink for one owned [`OutputElement`] per row.
pub struct ElementSink<T> {
    values: Vec<T>,
}

impl<T: OutputElement> OutputSink for ElementSink<T> {
    const SUPPORTS_SKIPPED_ROWS: bool = true;

    type Rows<'a> = &'a mut [T];
    type Row<'a> = &'a mut T;

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(T::element_dtype())
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        // `vec![placeholder; rows]` rather than `resize_with(rows, placeholder)`: the former hands
        // a zeroable placeholder (every primitive, `false`) straight to `alloc_zeroed`, while the
        // latter always writes one element at a time. Only branch-and-skip ever reads a
        // placeholder back, so on the dense and filter paths that write is pure waste.
        Ok(Self {
            values: vec![T::placeholder(); rows],
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        &mut self.values
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    fn finish(self, _error: DeferredError) -> VortexResult<ArrayRef> {
        Ok(T::build(self.values))
    }
}
