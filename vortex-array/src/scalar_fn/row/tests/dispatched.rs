// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for row functions that choose their element types per batch.

use vortex_error::vortex_ensure;

use super::*;
use crate::match_each_integer_ptype;

#[derive(Clone)]
struct Max;

impl RowFn for Max {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.int_max");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        let DType::Primitive(ptype, _) = args[0] else {
            vortex_bail!("int_max requires primitive inputs, got {}", args[0]);
        };
        vortex_ensure!(
            ptype.is_int(),
            "int_max requires integer inputs, got {ptype}"
        );

        match_each_integer_ptype!(ptype, |T| {
            visitor.visit_prepared_into::<(T, T), ElementSink<T>, _, _>(
                |_| (),
                |&(), (a, b), output| *output = a.max(b),
            )
        })
    }
}

#[rstest]
#[case::i16(buffer![1i16, 9, 3].into_array(), buffer![4i16, 2, 3].into_array(), buffer![4i16, 9, 3].into_array())]
#[case::i64(buffer![1i64, 9, 3].into_array(), buffer![4i64, 2, 3].into_array(), buffer![4i64, 9, 3].into_array())]
#[case::u8(buffer![1u8, 9, 3].into_array(), buffer![4u8, 2, 3].into_array(), buffer![4u8, 9, 3].into_array())]
fn dispatches_at_each_integer_width(
    #[case] lhs: ArrayRef,
    #[case] rhs: ArrayRef,
    #[case] expected: ArrayRef,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();

    let result = apply(Max, [lhs, rhs], &mut ctx)?;

    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn rejects_a_float_width() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![1.0f64].into_array();
    let rhs = buffer![2.0f64].into_array();

    let error = apply(Max, [lhs, rhs], &mut ctx)
        .expect_err("a float width must be rejected at construction");

    assert!(
        error.to_string().contains("integer inputs"),
        "unexpected error: {error}"
    );
}
