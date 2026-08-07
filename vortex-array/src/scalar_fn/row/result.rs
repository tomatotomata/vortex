// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! What a sink-writing row closure may return.

use std::ops::BitOrAssign;

use vortex_error::VortexResult;

mod private {
    pub trait Sealed {}
}

/// A value-dependent failure bit reduced across the whole row loop and handed to the output sink.
///
/// Unlike [`VortexResult`], this never exits the loop. It is for kernels such as checked addition
/// that can safely write a provisional value for every row and report any failure once at the end.
///
/// **The reduction is one byte wide on purpose.** It is OR-reduced once per row alongside the
/// kernel's own arithmetic, so a wider accumulator caps how many rows a vector of the reduction
/// covers, whatever the element width. Carrying the bit in an `i64` instead cost the primitive
/// `Mul` kernel 3.1x at `i8`, 1.9x at `i16` and 1.2x at `i32`, and nothing at `i64` where the two
/// widths already agree (`binary_ops`, 65536 rows, divan fastest of 100 samples, best of two runs,
/// Apple M4 Max).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferredError(bool);

impl DeferredError {
    /// Record whether this row encountered an error.
    pub const fn new(failed: bool) -> Self {
        Self(failed)
    }

    /// Whether any row accumulated into this value failed.
    pub const fn occurred(self) -> bool {
        self.0
    }
}

impl BitOrAssign for DeferredError {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// What a row computation that _writes_ into an [`OutputSink`](crate::scalar_fn::OutputSink) may
/// produce: nothing, an early [`VortexResult`] error, or non-branching failure evidence.
///
/// The value is already in the sink by the time the closure returns, so the only thing left to
/// report is failure.
///
/// [`Accumulated`](Self::Accumulated) is the word the executor OR-reduces in a **local**, which is
/// what keeps the reduction in a register and the row loop vectorizable. It exists so that evidence
/// can be wider than one bit when narrowing it per row would cost more than carrying it: unsigned
/// multiplication hands back the discarded high half of its product, because comparing that half
/// against zero per row is what LLVM folds into `llvm.umul.with.overflow`, which has no vector form.
/// **The word must be no wider than the element**, or the reduction, rather than the arithmetic,
/// bounds how many rows a vector covers.
///
/// The sink never sees this. It is handed a plain [`DeferredError`] once, after the loop.
///
/// This trait is framework-only. Row functions choose one of the supplied return forms; custom
/// output representation belongs in [`OutputSink`](crate::scalar_fn::OutputSink).
pub trait SinkResult: 'static + private::Sealed {
    /// The word this result reduces into, kept in a loop-local by the executor.
    type Accumulated: 'static + Copy + Default;

    /// Whether this return type can carry an error.
    const FALLIBLE: bool;

    /// Whether this result carries non-branching failure evidence for the sink.
    const DEFERRED: bool;

    /// Merge this row's outcome into the batch-wide reduction.
    fn accumulate(self, accumulated: &mut Self::Accumulated) -> VortexResult<()>;

    /// Whether the finished reduction means some row failed.
    fn occurred(accumulated: Self::Accumulated) -> bool;
}

impl private::Sealed for () {}

impl SinkResult for () {
    type Accumulated = ();

    const FALLIBLE: bool = false;
    const DEFERRED: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        Ok(())
    }

    fn occurred(_accumulated: ()) -> bool {
        false
    }
}

impl private::Sealed for VortexResult<()> {}

impl SinkResult for VortexResult<()> {
    type Accumulated = ();

    const FALLIBLE: bool = true;
    const DEFERRED: bool = false;

    fn accumulate(self, _accumulated: &mut ()) -> VortexResult<()> {
        self
    }

    fn occurred(_accumulated: ()) -> bool {
        false
    }
}

/// The evidence widths a row closure may reduce. `bool` is the ordinary answer; the unsigned
/// integers exist for a kernel whose per-row comparison would cost it its vectorization.
macro_rules! impl_sink_result_word {
    ($($word:ty),+ $(,)?) => {
        $(
            impl private::Sealed for $word {}

            impl SinkResult for $word {
                type Accumulated = $word;

                const FALLIBLE: bool = false;
                const DEFERRED: bool = true;

                fn accumulate(self, accumulated: &mut $word) -> VortexResult<()> {
                    *accumulated |= self;
                    Ok(())
                }

                fn occurred(accumulated: $word) -> bool {
                    accumulated != <$word>::default()
                }
            }
        )+
    };
}

impl_sink_result_word!(bool, u8, u16, u32, u64);

#[cfg(test)]
mod tests {
    use super::DeferredError;

    #[test]
    fn one_failing_row_is_enough() {
        let mut error = DeferredError::default();
        assert!(!error.occurred());

        error |= DeferredError::new(false);
        assert!(!error.occurred());

        error |= DeferredError::new(true);
        assert!(error.occurred());

        error |= DeferredError::new(false);
        assert!(error.occurred());
    }
}
