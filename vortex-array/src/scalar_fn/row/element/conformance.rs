// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A shared conformance check every [`InputElement`] should be run through.

use std::hint::black_box;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::dtype::DType;
use crate::scalar_fn::InputElement;

/// Assert that `E` honors its [`InputElement`] contract over `array`, and rejects `rejected_dtype`.
///
/// The part worth checking mechanically is [`InputElement::DENSE_SAFE`]. An element claiming it will
/// be read at rows that are *null*, where an array guarantees nothing about the payload, and getting
/// the `const` wrong is either an out-of-bounds panic in production (the failure mode of
/// [#9090](https://github.com/vortex-data/vortex/issues/9090)) or an unnecessary valid-only
/// execution path. Nothing else verifies it, since the framework reads the `const` rather than
/// testing the claim.
///
/// So `array` **must** contain at least one null row, and its payload behind those nulls **must** be
/// deliberately extreme rather than zeroed, or the check passes vacuously. Build that safely by
/// putting the extreme values in the array first and masking those rows afterwards, as the callers of
/// this function do.
///
/// What this cannot check: [`DECODE_FALLIBLE`](InputElement::DECODE_FALLIBLE), which needs data that
/// is legal but malformed, and whether `validate` accepts everything it *should*, since only the
/// element knows its full dtype domain. Pass one representative rejection.
#[track_caller]
pub fn assert_element_conforms<E: InputElement>(
    array: ArrayRef,
    rejected_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let dtype = array.dtype().clone();
    E::validate(&dtype)?;

    assert!(
        E::validate(rejected_dtype).is_err(),
        "element accepted {rejected_dtype}, which it was expected to reject",
    );

    let len = array.len();
    let valid = array.validity()?.execute_mask(len, ctx)?;
    assert!(
        !valid.all_true(),
        "conformance needs a null row to read behind, but every row of the {dtype} input is valid",
    );

    let column = E::decode(array, ctx)?;
    let varying = E::varying(&column);
    assert_eq!(
        E::varying_len(&varying),
        len,
        "varying element view changed the decoded row count",
    );

    // The claim under test. Reading a null row may yield garbage, but it must not fault, so an
    // element that secretly follows a per-row offset panics here instead of in production.
    if E::DENSE_SAFE {
        for index in 0..len {
            black_box(E::get(&column, index));
            black_box(E::get_varying(&varying, index));
        }
    } else {
        for index in 0..len {
            if valid.value(index) {
                black_box(E::get(&column, index));
                black_box(E::get_varying(&varying, index));
            }
        }
    }

    Ok(())
}
