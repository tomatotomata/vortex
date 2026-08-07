// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests for fallible input decoding and its effect on execution strategy.

use super::*;

/// Stands in for an element that _parses_ its bytes, like a WKB geometry: malformed bytes in a
/// valid row are a domain error, so decoding can fail on otherwise legal input.
struct ParsedBytes;

impl InputElement for ParsedBytes {
    type Column = VarBinViewArray;
    type Varying<'a> = &'a VarBinViewArray;
    type Elem<'a> = usize;

    const DENSE_SAFE: bool = true;
    const DECODE_FALLIBLE: bool = true;

    fn validate(_dtype: &DType) -> VortexResult<()> {
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        array.execute::<VarBinViewArray>(ctx)
    }

    fn get(column: &Self::Column, index: usize) -> usize {
        column.views()[index].len() as usize
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        column
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        column.len()
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> usize
    where
        Self: 'a,
    {
        Self::get(column, index)
    }
}

/// Its row computation is total; only the decode can fail.
#[derive(Clone)]
struct TotalKernelOverParsedInput;

impl RowFn for TotalKernelOverParsedInput {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.total_over_parsed");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(ParsedBytes,), ElementSink<u64>, _, _>(
            |_| (),
            |&(), (len,), output| *output = len as u64,
        )
    }
}

/// The row closure is infallible, so reading only the kernel declaration would report
/// `false` and let dict pushdown speculatively evaluate the parse over unreferenced values.
#[test]
fn a_fallible_decode_makes_the_function_fallible() {
    assert!(ScalarFnVTable::is_fallible(
        &TotalKernelOverParsedInput,
        &EmptyOptions
    ));
}

/// And it must not run densely: rows behind nulls would be parsed too.
#[test]
fn a_fallible_decode_forces_filtering() {
    assert_eq!(
        policy(
            &TotalKernelOverParsedInput,
            &[DType::Binary(Nullability::Nullable)]
        ),
        RowPolicy::ValidOnly {
            filtered_decode_cost: 0
        }
    );
}
