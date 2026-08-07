// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for preparing batch-constant state once before the row loop.

use std::cell::Cell;

use super::*;
use crate::validity::Validity;

thread_local! {
    /// Which operands the last `prepare` saw as constant, as a bitmask (bit 0 for `x`, bit 1
    /// for `y`). Thread-local rather than a process global so concurrent tests in one process
    /// (plain `cargo test`) cannot race it; execution runs on the calling thread.
    static SEEN_CONSTANTS: Cell<u8> = const { Cell::new(u8::MAX) };
}

/// `sqrt(x^2 + y^2)` through [`RowVisitor::visit_prepared_into`]: the square of any constant
/// operand is hoisted out of the row loop, and recorded in [`SEEN_CONSTANTS`].
#[derive(Clone)]
struct PreparedHypot;

impl RowFn for PreparedHypot {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["x", "y"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.prepared_hypot");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(f64, f64), ElementSink<f64>, _, _>(
            |(x, y)| {
                SEEN_CONSTANTS.set(u8::from(x.is_some()) | (u8::from(y.is_some()) << 1));
                (x.map(|x| x * x), y.map(|y| y * y))
            },
            |&(x_sq, y_sq), (x, y), output| {
                *output = (x_sq.unwrap_or(x * x) + y_sq.unwrap_or(y * y)).sqrt();
            },
        )
    }
}

/// A constant operand reaches `prepare` as `Some`, and the result is identical to the same
/// value expanded into a full column, which reaches `prepare` as `None`.
#[test]
fn a_constant_operand_matches_its_expanded_column() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0, 8.0].into_array();

    let constant = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();
    let from_constant = apply(PreparedHypot, [x.clone(), constant], &mut ctx)?;
    assert_eq!(SEEN_CONSTANTS.get(), 0b10);

    let expanded = buffer![4.0f64, 4.0, 4.0].into_array();
    let from_expanded = apply(PreparedHypot, [x, expanded], &mut ctx)?;
    assert_eq!(SEEN_CONSTANTS.get(), 0b00);

    assert_arrays_eq!(from_constant, from_expanded, &mut ctx);
    Ok(())
}

/// A masked constant (the same value in every row, some rows null, how the compressor spells
/// an all-same-with-nulls chunk) is a batch constant too: the wrapper carries only validity,
/// which the lifting owns, so `prepare` sees the child's value and the null rows stay
/// null in the result.
#[test]
fn a_masked_constant_operand_is_seen_as_constant() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0, 8.0].into_array();

    let masked_constant = MaskedArray::try_new(
        ConstantArray::new(Scalar::from(4.0f64), 3).into_array(),
        Validity::from_iter([true, false, true]),
    )?
    .into_array();
    let result = apply(PreparedHypot, [x, masked_constant], &mut ctx)?;

    assert_eq!(SEEN_CONSTANTS.get(), 0b10);
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
        &mut ctx
    );
    Ok(())
}

/// With no constant operand every `ConstElems` slot is `None` and the loop computes exactly
/// what unit preparation would.
#[test]
fn all_varying_operands_prepare_nothing() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0].into_array();
    let y = buffer![4.0f64, 12.0].into_array();

    let result = apply(PreparedHypot, [x, y], &mut ctx)?;

    assert_eq!(SEEN_CONSTANTS.get(), 0b00);
    assert_arrays_eq!(result, PrimitiveArray::from_iter([5.0f64, 13.0]), &mut ctx);
    Ok(())
}

/// Two constant operands are folded to a single-row execution by the lifting, and that
/// row still goes through `prepare`, seeing both constants.
#[test]
fn all_constant_operands_fold_and_still_prepare() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = ConstantArray::new(Scalar::from(3.0f64), 4).into_array();
    let y = ConstantArray::new(Scalar::from(4.0f64), 4).into_array();

    let result = apply(PreparedHypot, [x, y], &mut ctx)?;

    assert_eq!(SEEN_CONSTANTS.get(), 0b11);
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([5.0f64, 5.0, 5.0, 5.0]),
        &mut ctx
    );
    Ok(())
}

/// Null rows pass through the prepared path exactly as through unit preparation.
#[test]
fn nulls_propagate_through_the_prepared_path() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = PrimitiveArray::from_option_iter([Some(3.0f64), None, Some(8.0)]).into_array();
    let y = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();

    let result = apply(PreparedHypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
        &mut ctx
    );
    Ok(())
}
