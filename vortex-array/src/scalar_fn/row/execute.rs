// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The pieces every row function is built from, whatever its `dispatch` chooses.
//!
//! These back the blanket impls in [`row_fn`](super::row_fn) and are deliberately not public:
//! [`RowFn`](crate::scalar_fn::RowFn) is the abstraction, these are its internals.

use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::DeferredError;
use crate::scalar_fn::ElementTuple;
use crate::scalar_fn::ExecutionArgs;
use crate::scalar_fn::OutputSink;
use crate::scalar_fn::SinkResult;

/// The value path out of a row executor, keeping a deferred row error distinct from structural
/// execution errors so nullable lifting retries only the former.
pub(super) enum RowExecution {
    /// A successfully built output column.
    Output(ArrayRef),

    /// A batch-wide row error that nullable lifting may retry over only the valid rows.
    DeferredError(VortexError),
}

impl RowExecution {
    /// Return the output or surface its deferred row error.
    pub(super) fn into_result(self) -> VortexResult<ArrayRef> {
        match self {
            Self::Output(output) => Ok(output),
            Self::DeferredError(error) => Err(error),
        }
    }
}

/// Validate the input dtypes of a sink-writing row function and return the dtype its sink builds.
///
/// The output dtype may be a function of the inputs. A sink can also own a batch-wide builder, such
/// as the shared byte and view buffers of a future string transform.
pub(super) fn validate_row_sink<A: ElementTuple, S: OutputSink>(
    args: &[DType],
) -> VortexResult<DType> {
    A::validate(args)?;
    let dtype = S::sink_dtype(args)?;
    vortex_ensure!(
        !dtype.is_nullable(),
        "row output sinks must declare a non-nullable dtype, got {dtype}",
    );
    Ok(dtype)
}

/// Decode every input column once, allocate the sink once, then write one row at a time.
///
/// The sink lives here rather than in the closure, so `apply` stays [`Fn`] and the loop keeps the
/// unconditional shape that lets it vectorize. Monomorphic in `A`, `S` and `R`, so `apply` and
/// [`OutputSink::row`] both inline.
pub(super) fn execute_row_sink_prepared<A: ElementTuple, P, S: OutputSink, R: SinkResult>(
    args: &dyn ExecutionArgs,
    sink_dtype: &DType,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(A::ConstElems<'_>) -> P,
    apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
) -> VortexResult<RowExecution> {
    let row_count = args.row_count();
    let mut sink = S::with_capacity(row_count, sink_dtype)?;
    let columns = A::decode(args, ctx)?;
    let state = prepare(A::constants(&columns));
    let mut accumulated = R::Accumulated::default();

    {
        let mut rows = sink.rows();
        vortex_ensure!(
            S::row_count_matches(&rows, row_count),
            "the output sink does not address exactly {row_count} rows",
        );

        if let Some(varying) = A::varying(&columns) {
            vortex_ensure!(
                A::varying_len_matches(&varying, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );

            for index in 0..row_count {
                apply(
                    &state,
                    A::get_varying(&varying, index),
                    S::row(&mut rows, index),
                )
                .accumulate(&mut accumulated)?;
            }
        } else {
            vortex_ensure!(
                A::decoded_lens_match(&columns, row_count),
                "a decoded row input does not address exactly {row_count} rows",
            );

            for index in 0..row_count {
                apply(&state, A::get(&columns, index), S::row(&mut rows, index))
                    .accumulate(&mut accumulated)?;
            }
        }
    }

    finish_sink(sink, DeferredError::new(R::occurred(accumulated)))
}

/// Run a prepared sink over only the rows set in `valid`, or decline when the sink cannot skip.
pub(super) fn execute_row_sink_branch<A: ElementTuple, P, S: OutputSink, R: SinkResult>(
    args: &dyn ExecutionArgs,
    sink_dtype: &DType,
    valid: &Mask,
    ctx: &mut ExecutionCtx,
    prepare: impl FnOnce(A::ConstElems<'_>) -> P,
    apply: impl Fn(&P, A::Elems<'_>, S::Row<'_>) -> R,
) -> VortexResult<Option<RowExecution>> {
    if !S::SUPPORTS_SKIPPED_ROWS {
        return Ok(None);
    }

    let Some(columns) = A::decode_null_tolerant(args, ctx)? else {
        return Ok(None);
    };
    let state = prepare(A::constants(&columns));
    let row_count = args.row_count();
    let mut sink = S::with_capacity(row_count, sink_dtype)?;
    let mut accumulated = R::Accumulated::default();

    let AllOr::Some(valid) = valid.bit_buffer() else {
        vortex_bail!("execute_row_sink_branch requires a mixed mask");
    };

    {
        let mut rows = sink.rows();
        vortex_ensure!(
            S::row_count_matches(&rows, row_count),
            "the output sink does not address exactly {row_count} rows",
        );

        let varying = A::varying(&columns);
        let lens_match = match &varying {
            Some(varying) => A::varying_len_matches(varying, row_count),
            None => A::decoded_lens_match(&columns, row_count),
        };
        vortex_ensure!(
            lens_match,
            "a decoded row input does not address exactly {row_count} rows",
        );

        let mut error = None;
        valid.for_each_set_index(|index| {
            if error.is_some() {
                return;
            }

            let result = match &varying {
                Some(varying) => apply(
                    &state,
                    A::get_varying(varying, index),
                    S::row(&mut rows, index),
                ),
                None => apply(&state, A::get(&columns, index), S::row(&mut rows, index)),
            };
            if let Err(err) = result.accumulate(&mut accumulated) {
                error = Some(err);
            }
        });

        if let Some(error) = error {
            return Err(error);
        }
    }

    finish_sink(sink, DeferredError::new(R::occurred(accumulated))).map(Some)
}

/// Finish a sink while preserving whether its error came from the deferred row accumulator.
fn finish_sink<S: OutputSink>(
    sink: S,
    deferred_error: DeferredError,
) -> VortexResult<RowExecution> {
    match sink.finish(deferred_error) {
        Ok(output) => Ok(RowExecution::Output(output)),
        Err(error) if deferred_error.occurred() => Ok(RowExecution::DeferredError(error)),
        Err(error) => Err(error),
    }
}
