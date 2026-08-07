// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defining scalar functions one row at a time.
//!
//! This is the derived way to write a scalar function, and the right default for a kernel that has
//! to read every row anyway. See [choosing a trait](crate::scalar_fn#choosing-a-trait) for when to
//! drop to [`ScalarFnVTable`](crate::scalar_fn::ScalarFnVTable) instead.
//!
//! [`RowFn`] names its arguments and provides a [`dispatch`](RowFn::dispatch) that picks the
//! concrete element and sink types for a batch. Everything else (dtype checks, output dtype, null
//! handling, constants, and validity) is derived from that dispatch.
//!
//! When the element types are fixed, `dispatch` is a single visit at those types. When one function
//! ID has to cover several (`l2_norm` accepts `f16`, `f32` and `f64` columns), `dispatch` matches on
//! the input dtypes and visits at the chosen width. Kernel fallibility is declared separately
//! because callers need it before dispatch.
//!
//! [`RowFn`] does not say how a row is _stored_, which is the element's job: `vortex-tensor` adds a
//! `TensorRow<T>` [`InputElement`] and writes ordinary kernels over it.
//!
//! Output always goes through [`RowVisitor::visit_prepared_into`]. [`ElementSink`] covers one owned
//! [`OutputElement`] per row; custom [`OutputSink`] implementations cover runtime-shaped rows. The
//! prepare closure sees every batch-constant input and returns shared state for the row loop. Pass
//! `|_| ()` when there is nothing to prepare.
//!
//! A kernel that can safely write a provisional value uses [`DeferredError`] instead of returning
//! a per-row result. The executor vector-reduces those bits and hands one batch-wide error to the
//! sink. With nullable fixed-width inputs it runs densely and retries only valid rows on the cold
//! error path.
//!
//! Null handling is derived and executed by the [lifting](lift), never by the row closure, which
//! only ever computes rows valid in every argument. A batch with a mixed validity mask executes by
//! one of two strategies, selected per batch: _branch-and-skip_ (decode the unfiltered columns
//! null-tolerantly via [`InputElement::decode_null_tolerant`], compute only the valid rows a word
//! of the mask at a time, mask the result) whenever it can, and _filter_ (shrink every input to
//! the surviving rows, compute, scatter back) when an argument has no null-tolerant decode for its
//! array or when a per-row decode makes filtering cheaper at sparse validity. Authors do nothing;
//! an element whose decode does expensive per-row work reports that work through
//! [`InputElement::FILTERED_DECODE_COST`]. The costs of all arguments are added together so the
//! batch selector can distinguish one expensive decode from several. A sink opts into
//! branch-and-skip with [`OutputSink::SUPPORTS_SKIPPED_ROWS`].

mod element;
pub use element::ElementTuple;
pub use element::InputElement;
pub use element::OutputElement;
#[cfg(any(test, feature = "_test-harness"))]
pub use element::assert_element_conforms;

mod result;
pub use result::DeferredError;
pub use result::SinkResult;

mod sink;
pub use sink::ElementSink;
pub use sink::OutputSink;

mod execute;

mod lift;
#[cfg(any(test, feature = "_test-harness"))]
pub use lift::NullStrategy;

mod row_fn;
pub use row_fn::RowFn;
pub use row_fn::RowVisitor;

mod vtable;
#[cfg(any(test, feature = "_test-harness"))]
pub use vtable::execute_row_fn_with_strategy;

#[cfg(test)]
mod tests;
