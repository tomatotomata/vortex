// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests that row output dtypes cannot introduce their own nulls.

use super::*;
use crate::dtype::Nullability;
use crate::dtype::PType;

#[derive(Clone)]
struct NullableI64(i64);

impl OutputElement for NullableI64 {
    fn element_dtype() -> DType {
        DType::Primitive(PType::I64, Nullability::Nullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        PrimitiveArray::from_option_iter(values.into_iter().map(|value| Some(value.0))).into_array()
    }

    fn placeholder() -> Self {
        Self(0)
    }
}

struct NullableSink(usize);

impl OutputSink for NullableSink {
    type Rows<'a> = usize;
    type Row<'a> = ();

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::Primitive(PType::I64, Nullability::Nullable))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self(rows))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        *rows == row_count
    }

    fn row<'a>(_rows: &'a mut Self::Rows<'_>, _index: usize) -> Self::Row<'a> {}

    fn finish(self, _error: DeferredError) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::from_option_iter(Vec::<Option<i64>>::new()).into_array())
    }
}

#[derive(Clone)]
struct NullableElementFn;

impl RowFn for NullableElementFn {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.nullable_element");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), ElementSink<NullableI64>, _, _>(
            |_| (),
            |&(), (value,), output| *output = NullableI64(value),
        )
    }
}

#[derive(Clone)]
struct NullableSinkFn;

impl RowFn for NullableSinkFn {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.nullable_sink");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), NullableSink, _, _>(|_| (), |&(), _, ()| {})
    }
}

#[test]
fn nullable_element_dtype_is_rejected() {
    let input = DType::Primitive(PType::I64, Nullability::NonNullable);
    let error =
        ScalarFnVTable::return_dtype(&NullableElementFn, &EmptyOptions, &[input]).unwrap_err();

    assert!(error.to_string().contains("non-nullable dtype"), "{error}");
}

#[test]
fn nullable_sink_dtype_is_rejected() {
    let input = DType::Primitive(PType::I64, Nullability::NonNullable);
    let error = ScalarFnVTable::return_dtype(&NullableSinkFn, &EmptyOptions, &[input]).unwrap_err();

    assert!(error.to_string().contains("non-nullable dtype"), "{error}");
}
