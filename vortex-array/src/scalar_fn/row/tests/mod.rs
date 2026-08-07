// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! End-to-end tests for row function execution.

use rstest::rstest;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::registry::CachedId;

use super::lift::RowPolicy;
use super::vtable::row_policy;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::MaskedArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::scalar_fn::ScalarFnFactoryExt;
use crate::arrays::varbinview::BinaryView;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::expr::root;
use crate::scalar::Scalar;
use crate::scalar_fn::*;

mod conformance;
mod constant_operands;
mod decode_fallibility;
mod dispatched;
mod lifting;
mod null_strategies;
mod nullable_outputs;
mod prepared;
mod sink;

/// Builds `scalar_fn` over `args` and executes it end to end, which is what every test below does.
fn apply<F: RowFn<Options = EmptyOptions>>(
    scalar_fn: F,
    args: impl IntoIterator<Item = ArrayRef>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let args = args.into_iter().collect::<Vec<_>>();
    let rows = args.first().map_or(0, |arg| arg.len());

    Ok(scalar_fn
        .try_new_array(rows, EmptyOptions, args)?
        .execute::<Canonical>(ctx)?
        .into_array())
}

/// A binary row function over fixed primitive types: `hypot(x, y)`.
#[derive(Clone)]
struct Hypot;

impl RowFn for Hypot {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["x", "y"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.hypot");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(f64, f64), ElementSink<f64>, _, _>(
            |_| (),
            |&(), (x, y), output| *output = x.hypot(y),
        )
    }
}

/// A byte-string element that resolves each row's view into a data buffer, which is only
/// meaningful for a valid row.
///
/// This is the crate's only non-dense-safe element, so it exercises valid-only execution,
/// branch-and-skip, and the agreement between the two strategies. It lives here rather than beside
/// the framework because no production row function reads bytes yet.
struct TestBytes;

/// The canonical views array plus its resolved data buffers.
struct TestBytesColumn {
    array: VarBinViewArray,
    buffers: Vec<ByteBuffer>,
}

/// Resolve one view, which is either inlined or an offset into `buffers`.
fn read_view<'a>(view: &'a BinaryView, buffers: &'a [ByteBuffer]) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view = view.as_view();
        &buffers[view.buffer_index as usize].as_slice()[view.as_range()]
    }
}

impl InputElement for TestBytes {
    type Column = TestBytesColumn;
    // The views slice, not the array: `VarBinViewArray::views` resolves a host buffer and its
    // `vortex_expect` is a side effect the optimizer cannot hoist out of the row loop.
    type Varying<'a> = (&'a [BinaryView], &'a [ByteBuffer]);
    type Elem<'a> = &'a [u8];

    const DENSE_SAFE: bool = false;
    const DECODE_FALLIBLE: bool = false;

    fn validate(dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "expected a Utf8 or Binary column, got {dtype}",
        );
        Ok(())
    }

    fn decode(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self::Column> {
        let array = array.execute::<VarBinViewArray>(ctx)?;
        let buffers = (0..array.data_buffers().len())
            .map(|idx| array.buffer(idx).clone())
            .collect();
        Ok(TestBytesColumn { array, buffers })
    }

    fn decode_null_tolerant(
        array: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self::Column>> {
        Self::decode(array, ctx).map(Some)
    }

    fn get(column: &Self::Column, index: usize) -> &[u8] {
        read_view(&column.array.views()[index], &column.buffers)
    }

    fn varying(column: &Self::Column) -> Self::Varying<'_> {
        (column.array.views(), &column.buffers)
    }

    fn varying_len(column: &Self::Varying<'_>) -> usize {
        column.0.len()
    }

    fn get_varying<'a>(column: &Self::Varying<'a>, index: usize) -> &'a [u8]
    where
        Self: 'a,
    {
        read_view(&column.0[index], column.1)
    }
}

impl OutputElement for String {
    fn element_dtype() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    fn build(values: Vec<Self>) -> ArrayRef {
        VarBinViewArray::from_iter_str(values).into_array()
    }

    fn placeholder() -> Self {
        String::new()
    }
}

/// A unary row function over strings: uppercased text, exercising [`TestBytes`] input and
/// [`String`] output.
#[derive(Clone)]
struct Shout;

impl RowFn for Shout {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.shout");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(TestBytes,), ElementSink<String>, _, _>(
            |_| (),
            |&(), (text,), output| {
                *output = String::from_utf8_lossy(text).to_uppercase();
            },
        )
    }
}

/// A fallible row function: integer division, undefined at a zero divisor.
#[derive(Clone)]
struct CheckedDiv;

impl RowFn for CheckedDiv {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.checked_div");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| {
                if rhs == 0 {
                    vortex_bail!("division by zero");
                }
                *output = lhs / rhs;
                Ok(())
            },
        )
    }
}

#[test]
fn hypot_columns() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = buffer![3.0f64, 5.0].into_array();
    let y = buffer![4.0f64, 12.0].into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([5.0f64, 13.0]), &mut ctx);
    Ok(())
}

#[test]
fn hypot_propagates_nulls_and_constants() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let x = PrimitiveArray::from_option_iter([Some(3.0f64), None, Some(8.0)]).into_array();
    let y = ConstantArray::new(Scalar::from(4.0f64), 3).into_array();

    let result = apply(Hypot, [x, y], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5.0f64), None, Some((80.0f64).sqrt())]),
        &mut ctx
    );
    Ok(())
}

#[test]
fn shout_strings() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let input =
        VarBinViewArray::from_iter_nullable_str([Some("hello"), None, Some("Vortex")]).into_array();

    let result = apply(Shout, [input], &mut ctx)?;

    let expected =
        VarBinViewArray::from_iter_nullable_str([Some("HELLO"), None, Some("VORTEX")]).into_array();
    assert_arrays_eq!(result, expected, &mut ctx);
    Ok(())
}

#[test]
fn display_names_the_function_id() {
    let expr = Hypot.new_expr(EmptyOptions, [root(), root()]);
    assert_eq!(expr.to_string(), "vortex.test.hypot($, $)");
}

#[derive(Clone)]
struct WrongLength;

impl RowFn for WrongLength {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["input"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.wrong_length");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64,), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (value,), output| *output = value,
        )
    }

    fn reduce_encoded(
        &self,
        _options: &Self::Options,
        _args: &[ArrayRef],
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(PrimitiveArray::from_iter([0i64]).into_array()))
    }
}

#[test]
fn kernel_result_length_is_validated() {
    let mut ctx = array_session().create_execution_ctx();
    let input = buffer![1i64, 2, 3].into_array();

    let error = apply(WrongLength, [input], &mut ctx).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("produced 1 rows for 3 input rows"),
        "{error}"
    );
}

#[derive(Clone)]
struct FortyTwo;

impl RowFn for FortyTwo {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &[];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.forty_two");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(), ElementSink<i64>, _, _>(
            |()| (),
            |&(), (), output| *output = 42,
        )
    }
}

#[test]
fn nullary_row_fn_executes_requested_rows() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = FortyTwo
        .try_new_array(3, EmptyOptions, [])?
        .execute::<Canonical>(&mut ctx)?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([42i64; 3]), &mut ctx);
    Ok(())
}

#[derive(Clone)]
struct SumFour;

impl RowFn for SumFour {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["a", "b", "c", "d"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.test.sum_four");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64, i64, i64), ElementSink<i64>, _, _>(
            |_| (),
            |&(), (a, b, c, d), output| *output = a + b + c + d,
        )
    }
}

#[test]
fn four_argument_row_fn_executes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = apply(
        SumFour,
        [
            buffer![1i64, 2].into_array(),
            buffer![10i64, 20].into_array(),
            buffer![100i64, 200].into_array(),
            buffer![1000i64, 2000].into_array(),
        ],
        &mut ctx,
    )?;

    assert_arrays_eq!(result, PrimitiveArray::from_iter([1111i64, 2222]), &mut ctx);
    Ok(())
}

#[test]
fn tuples_are_supported_through_arity_twelve() {
    type TwelveI64s = (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);

    assert_eq!(<() as ElementTuple>::ARITY, 0);
    assert_eq!(<TwelveI64s as ElementTuple>::ARITY, 12);
}

#[test]
fn kernel_flag_decides_fallibility() {
    assert!(!ScalarFnVTable::is_fallible(&Hypot, &EmptyOptions));
    assert!(ScalarFnVTable::is_fallible(&CheckedDiv, &EmptyOptions));
}

#[test]
fn fallible_apply_propagates_its_error() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = buffer![2i64, 0].into_array();

    let error = apply(CheckedDiv, [lhs, rhs], &mut ctx)
        .expect_err("a zero divisor must fail the execution");

    assert!(
        error.to_string().contains("division by zero"),
        "unexpected error: {error}"
    );
    Ok(())
}

/// The divisor's null slot holds a zero, which a dense pass would divide by. Filtering keeps the
/// fallible kernel away from it.
#[test]
fn fallible_apply_never_sees_rows_behind_nulls() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = buffer![10i64, 10].into_array();
    let rhs = PrimitiveArray::from_option_iter([Some(2i64), None]).into_array();

    let result = apply(CheckedDiv, [lhs, rhs], &mut ctx)?;

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(5i64), None]),
        &mut ctx
    );
    Ok(())
}

/// The internal nullable execution policy selected by a concrete dispatch.
fn policy<F: RowFn<Options = EmptyOptions>>(row_fn: &F, args: &[DType]) -> RowPolicy {
    row_policy(row_fn, &EmptyOptions, args).expect("test dispatch must produce a policy")
}

/// The function never declares a policy: the dispatched arguments and result decide it.
#[test]
fn null_handling_follows_from_args_and_fallibility() {
    // Primitive arguments, infallible: nothing behind a null row can fault.
    assert_eq!(
        policy(
            &Hypot,
            &[
                DType::Primitive(PType::F64, Nullability::NonNullable),
                DType::Primitive(PType::F64, Nullability::NonNullable),
            ]
        ),
        RowPolicy::Dense
    );
    // `TestBytes` resolves a view into a data buffer, which is only meaningful for valid rows.
    assert_eq!(
        policy(&Shout, &[DType::Utf8(Nullability::Nullable)]),
        RowPolicy::ValidOnly {
            filtered_decode_cost: 0
        }
    );
    // Fallible: a garbage row could raise an error of its own.
    assert_eq!(
        policy(
            &CheckedDiv,
            &[
                DType::Primitive(PType::I64, Nullability::NonNullable),
                DType::Primitive(PType::I64, Nullability::NonNullable),
            ]
        ),
        RowPolicy::ValidOnly {
            filtered_decode_cost: 0
        }
    );
}
