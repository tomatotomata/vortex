// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compares cheap primitive row functions with the specialized columnar implementation.

#![expect(clippy::unwrap_used)]

use std::mem::MaybeUninit;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::DeferredError;
use vortex_array::scalar_fn::ElementSink;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::OutputSink;
use vortex_array::scalar_fn::RowFn;
use vortex_array::scalar_fn::RowVisitor;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

const ROWS: usize = 65_536;

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

#[derive(Clone)]
struct RowWrappingAdd;

impl RowFn for RowWrappingAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_wrapping_add");
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
            |&(), (lhs, rhs), output| *output = lhs.wrapping_add(rhs),
        )
    }
}

#[derive(Clone)]
struct RowCheckedAdd;

struct CheckedAddSink {
    values: BufferMut<i64>,
    row_count: usize,
}

struct CheckedAddRows<'a> {
    values: &'a mut [MaybeUninit<i64>],
}

struct CheckedAddRow<'a> {
    value: &'a mut MaybeUninit<i64>,
}

impl CheckedAddRow<'_> {
    fn write(self, lhs: i64, rhs: i64) -> bool {
        let value = lhs.wrapping_add(rhs);
        let error = (lhs ^ value) & (rhs ^ value);
        self.value.write(value);
        error < 0
    }
}

impl OutputSink for CheckedAddSink {
    const ERRORS_ARE_DEFERRED: bool = true;

    type Rows<'a> = CheckedAddRows<'a>;
    type Row<'a> = CheckedAddRow<'a>;

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self {
            values: BufferMut::with_capacity(rows),
            row_count: rows,
        })
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        CheckedAddRows {
            values: &mut self.values.spare_capacity_mut()[..self.row_count],
        }
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.values.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        CheckedAddRow {
            value: &mut rows.values[index],
        }
    }

    fn finish(mut self, error: DeferredError) -> VortexResult<ArrayRef> {
        if error.occurred() {
            return Err(vortex_err!("integer overflow in row checked add"));
        }

        // SAFETY: dense execution writes every slot before `finish` is called. This sink does not
        // support branch-and-skip, and filtered execution allocates exactly one slot per valid row.
        unsafe { self.values.set_len(self.row_count) };
        Ok(PrimitiveArray::new(self.values.freeze(), Validity::NonNullable).into_array())
    }
}

impl RowFn for RowCheckedAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];
    const FALLIBLE: bool = true;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_checked_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64), CheckedAddSink, _, _>(
            |_| (),
            |&(), (lhs, rhs), output| output.write(lhs, rhs),
        )
    }
}

struct I64Sink(BufferMut<i64>);

impl OutputSink for I64Sink {
    type Rows<'a> = &'a mut [i64];
    type Row<'a> = &'a mut i64;

    fn sink_dtype(_args: &[DType]) -> VortexResult<DType> {
        Ok(DType::from(i64::PTYPE))
    }

    fn with_capacity(rows: usize, _dtype: &DType) -> VortexResult<Self> {
        Ok(Self(BufferMut::zeroed(rows)))
    }

    fn rows(&mut self) -> Self::Rows<'_> {
        self.0.as_mut_slice()
    }

    fn row_count_matches(rows: &Self::Rows<'_>, row_count: usize) -> bool {
        rows.len() == row_count
    }

    fn row<'a>(rows: &'a mut Self::Rows<'_>, index: usize) -> Self::Row<'a> {
        &mut rows[index]
    }

    fn finish(self, _error: DeferredError) -> VortexResult<ArrayRef> {
        Ok(PrimitiveArray::new(self.0.freeze(), Validity::NonNullable).into_array())
    }
}

#[derive(Clone)]
struct RowSinkWrappingAdd;

impl RowFn for RowSinkWrappingAdd {
    type Options = EmptyOptions;

    const ARG_NAMES: &'static [&'static str] = &["lhs", "rhs"];

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("bench.row_sink_wrapping_add");
        *ID
    }

    fn dispatch<V: RowVisitor>(
        &self,
        _options: &Self::Options,
        _args: &[DType],
        visitor: V,
    ) -> VortexResult<V::Out> {
        visitor.visit_prepared_into::<(i64, i64), I64Sink, _, _>(
            |_| (),
            |&(), (lhs, rhs), out| {
                *out = lhs.wrapping_add(rhs);
            },
        )
    }
}

fn inputs() -> (ArrayRef, ArrayRef) {
    let lhs = (0..ROWS)
        .map(|index| index as i64)
        .collect::<Buffer<_>>()
        .into_array();
    let rhs = (0..ROWS)
        .map(|index| (index % 17) as i64)
        .collect::<Buffer<_>>()
        .into_array();
    (lhs, rhs)
}

fn constant_inputs() -> (ArrayRef, ArrayRef) {
    let (lhs, _) = inputs();
    let rhs = ConstantArray::new(Scalar::from(7i64), ROWS).into_array();
    (lhs, rhs)
}

fn nullable_inputs() -> (ArrayRef, ArrayRef) {
    let lhs = PrimitiveArray::new(
        (0..ROWS).map(|index| index as i64).collect::<Buffer<_>>(),
        Validity::from_iter((0..ROWS).map(|index| !index.is_multiple_of(5))),
    )
    .into_array();
    let rhs = PrimitiveArray::new(
        (0..ROWS)
            .map(|index| (index % 17) as i64)
            .collect::<Buffer<_>>(),
        Validity::from_iter((0..ROWS).map(|index| !index.is_multiple_of(7))),
    )
    .into_array();
    (lhs, rhs)
}

fn bench_row_fn<F>(bencher: Bencher, row_fn: F)
where
    F: RowFn<Options = EmptyOptions>,
{
    bencher
        .with_inputs(inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            row_fn
                .clone()
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn specialized_checked_add(bencher: Bencher) {
    bencher
        .with_inputs(inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            lhs.binary(rhs, Operator::Add)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_wrapping_add(bencher: Bencher) {
    bench_row_fn(bencher, RowWrappingAdd);
}

#[divan::bench]
fn row_sink_wrapping_add(bencher: Bencher) {
    bench_row_fn(bencher, RowSinkWrappingAdd);
}

#[divan::bench]
fn handrolled_sink_wrapping_add(bencher: Bencher) {
    bencher
        .with_inputs(inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            let lhs = lhs
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_buffer::<i64>();
            let rhs = rhs
                .execute::<PrimitiveArray>(&mut ctx)
                .unwrap()
                .into_buffer::<i64>();
            let mut output = BufferMut::zeroed(ROWS);
            for ((out, lhs), rhs) in output
                .as_mut_slice()
                .iter_mut()
                .zip(lhs.as_slice())
                .zip(rhs.as_slice())
            {
                *out = lhs.wrapping_add(*rhs);
            }
            PrimitiveArray::new(output.freeze(), Validity::NonNullable).into_array()
        });
}

#[divan::bench]
fn row_checked_add(bencher: Bencher) {
    bench_row_fn(bencher, RowCheckedAdd);
}

#[divan::bench]
fn specialized_checked_add_constant(bencher: Bencher) {
    bencher
        .with_inputs(constant_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            lhs.binary(rhs, Operator::Add)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_wrapping_add_constant(bencher: Bencher) {
    bencher
        .with_inputs(constant_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            RowWrappingAdd
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_checked_add_constant(bencher: Bencher) {
    bencher
        .with_inputs(constant_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            RowCheckedAdd
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn specialized_checked_add_nullable(bencher: Bencher) {
    bencher
        .with_inputs(nullable_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            lhs.binary(rhs, Operator::Add)
                .unwrap()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_checked_add_nullable(bencher: Bencher) {
    bencher
        .with_inputs(nullable_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            RowCheckedAdd
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}

#[divan::bench]
fn row_wrapping_add_nullable(bencher: Bencher) {
    bencher
        .with_inputs(nullable_inputs)
        .bench_local_values(|(lhs, rhs)| {
            let mut ctx = SESSION.create_execution_ctx();
            RowWrappingAdd
                .try_new_array(ROWS, EmptyOptions, [lhs, rhs])
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap()
        });
}
