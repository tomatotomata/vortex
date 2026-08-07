<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Plan: fit the numeric binary operators onto `RowFn`

Working note, branch-only, like `SCALAR_FN_HANDOFF.md`. Written so this survives a conversation
compaction: everything needed to start is here, and nothing below depends on chat history.

## Where things stand

Branch `ct/row-fn`, at `4becc863ae` after the final API
simplification. Issues #9128, #9129, and #9130 match the implementation. The public-path benchmark
baseline from #9136 is now in the repository.

This document preserves the original spike plan and the measurements that answered it. The current
API has no witnesses, persistence is function-owned, executor-only helper traits are sealed, and
filtered decode cost is additive per input. Read the outcome and final API/codegen sections before
following an earlier step literally.

`byte_length` is no longer a row function, and `Bytes`/`BytesLen` are deleted. It measured 7.6-7.7x
slower than develop and is the case #9128 already excludes.

## Goal of this spike

Prove, or disprove, that the four arithmetic operators can move onto `RowFn` without changing the
`RowFn` API, without a second scalar function ID, and without touching serialization. Doing this
first is deliberate: it is the change most likely to force an API change, and discovering that after
tensor and geo are ported would mean reworking them.

## The design

`Binary` keeps everything and delegates only execution:

```rust
Operator::Add => ScalarFnVTable::execute(&NumericBinary, &NumericOperator::Add, args, ctx),
```

`NumericBinary` is a `RowFn` with `Options = NumericOperator` and `FALLIBLE = true`. It is **not**
registered as a public scalar function, so it needs no ID in the registry and appears in no
serialized expression. It is reached through the `ScalarFnVTable::execute` that the blanket impl
already provides.

Why this works, and each of these was verified against the code rather than assumed:

- **Nothing is lost.** `BooleanKernel` and `CompareKernel` exist with per-encoding pushdown; there is
  no `NumericKernel`. Unlike `not`, a numeric port gives up no encoding fast path.
- **The seam is already numeric-only.** All four arithmetic arms of `Binary::execute` funnel into
  `execute_numeric(lhs, rhs, NumericOperator, ctx)`, and `NumericOperator` is already its own enum in
  `crate::scalar`, so it is a ready-made `RowFn::Options`.
- **Fallibility is uniform.** `Binary::is_fallible` is false for the six comparisons plus `And`/`Or`
  and true for exactly the four arithmetic operators, so `FALLIBLE = true` on a numeric-only `RowFn`
  is exactly right. The options-independence of `RowFn::is_fallible` only bites when one function
  spans both families.
- **Strictness stays where it belongs.** `Binary::is_strict` is `!matches!(op, And | Or)` because
  Kleene `false AND null` is a valid `false`. `Binary` keeps owning that; `NumericBinary` never sees
  a boolean operator.
- **Decimal fits.** `OutputSink::sink_dtype(args)` sees the input dtypes, which is what
  `numeric_op_result_decimal_dtype(decimal_dtype, op)` needs.

## Steps

1. **Primitive path only, `Add` only.** A `NumericBinary` `RowFn` over `(T, T)` for one integer
   width, with a deferred-error sink that writes the wrapping sum and ORs an overflow bit. Delegate
   only `Operator::Add` from `Binary::execute` and leave the other three on `execute_numeric`.
   Success is: the existing `binary/numeric/tests.rs` suite passes unchanged.
2. **Widen to every primitive ptype**, through `match_each_native_ptype!` in `dispatch`. Confirm the
   compile-time witness check tolerates it, as it does for tensor widths.
3. **Add `Sub`, `Mul`, `Div`.** `Div` is the awkward one: see the risk below.
4. **Decide decimal.** Either a decimal input element plus a sink that carries the result precision
   and scale, or leave `DType::Decimal` on `execute_numeric` and delegate only the primitive path.
   Leaving it is a legitimate outcome for the spike and possibly for the first PR.
5. **Delete the replaced code** only once benchmarks agree, not before.

## Risks, in the order they are likely to bite

- **`Div` already has a per-type strategy.** `primitive.rs` carries `CHECKED_VALUE_LOOP` and
  `DIV_CHECKS_IN_VALUE_LOOP`, set per type, so division checking is not uniform. A single row closure
  may not express it, and `Div` may have to stay behind.
- **The existing implementation is tuned, not naive.** `checked.rs` has `checked_lanes` and
  `checked_apply_lanes` taking a `valid_rows: &Mask` and returning `Result<Buffer<T>, usize>` with the
  failing index. The port is replacing real engineering, so parity is not a given. This is the reason
  the CodSpeed gate on the `binary_ops` names from #9136 matters.
- **Two declarations of the result dtype must agree.** `Binary::return_dtype` is what the expression
  layer uses, while `reconcile_return` checks the kernel output against `NumericBinary`'s
  sink-derived dtype. Cover every operator and dtype pair with a test that asserts they match.
- **Error messages are part of the contract.** `primitive.rs` defines `ERROR` per operator, such as
  `"integer overflow in checked add"`, and `numeric/tests.rs` asserts on failures. The deferred-error
  sink reports once from `finish`, so the message must be preserved and the error must still be
  raised for the same inputs.
- **Overflow behind a null row must stay invisible.** `numeric/tests.rs` has
  `test_decimal_overflow_on_null_lane_ignored`. The lifting's deferred-error retry over valid rows is
  exactly this behavior, so the test should pass, but it is the first thing to check.

## Verification

```bash
cargo nextest run -p vortex-array
cargo clippy --all-targets --all-features -p vortex-array
cargo +nightly fmt --all
cargo test --doc -p vortex-array
```

The numeric suite specifically:

```bash
cargo nextest run -p vortex-array scalar_fn::fns::binary
```

Performance gate is CodSpeed on the stable `binary_ops` names from #9136. Locally, use
`cargo bench -p vortex-array --bench binary_ops` with two runs, fastest and median, machine stated.

## What this spike is not

Not a PR. Not a deletion of `execute_numeric`. Not decimal support unless step 4 turns out easy. The
output is an answer to "does this fit cleanly", plus whatever the answer implies for #9129's API.

## Outcome

It fits, with no change to the `RowFn` API and one change to the machinery.

Steps 1 through 3 landed together rather than in sequence: once the sink existed, widening it through
`match_each_native_ptype!` and adding the other three operators was the same code. Step 4 leaves
decimal on `execute_numeric_decimal`, which the delegation makes easy since `execute_numeric` still
owns the dtype split. Step 5 deleted the replaced primitive execution, which the measurements below
justify.

### What the design turned out to be

`Binary::execute` is untouched. `execute_numeric` keeps its validation, its error messages, its empty
short circuit, and its primitive/decimal split, and only `execute_numeric_primitive` changed: it
builds a `VecExecutionArgs` and calls `ScalarFnVTable::execute(&NumericBinary, &op, ..)`. Everything
the old implementation did around the arithmetic (decoding, the constant-operand collapse, the
all-constant fold, the null-constant short circuit, output allocation, nullability widening, masking,
and the valid-row retry after an overflow behind a null) is now the lifting's.

`NumericOperator` became the options type. `NumericBinary` is unregistered and deliberately has no
serialization implementation. Persistence now belongs to each `RowFn`, so reusing an options type
does not silently assign the helper a wire contract. `Binary` retains its existing ID and options
serialization, and only primitive execution delegates to `NumericBinary`.

Three things the old code carried are gone because the row framework removes the distinction they
existed for:

- `CHECKED_VALUE_LOOP` and `DIV_CHECKS_IN_VALUE_LOOP` chose between a split value/error scan and a
  one-pass early-exit kernel, because for integer division the split loop only added a second scan.
  A row kernel produces the value and the error bit in the same pass, so there is one loop shape and
  no choice to make. `div_i64` got 1.11x faster.
- `checked_apply_lanes` had no caller left. `checked_lanes` stays for decimal.
- `PrimitiveOperand` moved to `compare/primitive.rs`, its only remaining user.

### The machinery change: the reduction is a word the kernel chooses

`SinkResult` gained `Accumulated`, the word the executor OR-reduces in a loop-local. Two properties
of that reduction are load-bearing, and each was got wrong once before the numbers made it obvious.

- **Width no greater than the element.** `DeferredError` held an `i64`, which bounds how many rows a
  vector of the reduction covers whatever the element width. That cost `Mul` 3.5x at `i8`, 2.05x at
  `i16` and 1.28x at `i32`, and nothing at `i64` where the widths already agree.
- **It lives in a loop-local, not in the sink.** Holding the accumulator as a sink field, reached
  through a `&mut` for every row, is a loop-carried memory dependence. It cost the boolean kernels
  2.5x to 10x while leaving the three unsigned multiply kernels untouched.

Naming the word is also what lets multiplication report the discarded high half of its product
rather than a comparison, which is what recovers its vectorization. `OutputSink` is unchanged and no
sink names the word.

### Results

Against the hand-written kernels, divan medians, best of two runs, 65536 rows, Apple M4 Max, with
the decimal, boolean and comparison benchmarks held as controls and moving under 2%:

| benchmark | hand-written | row framework | |
| --- | --- | --- | --- |
| `mul_u8_nonnull` | 22.91 us | 1.854 us | 12.4x faster |
| `mul_u16_nonnull` | 22.20 us | 3.791 us | 5.9x faster |
| `mul_u32_nonnull` | 24.62 us | 7.124 us | 3.5x faster |
| `div_i64_nonnull` | 40.41 us | 34.83 us | 1.16x faster |
| `mul_i64_nonnull` | 27.37 us | 28.66 us | 1.05x slower |
| `mul_i32_constant` | 7.583 us | 8.041 us | 1.06x slower |

Everything else lands within 3%, which is inside this host's drift between sessions. The unsigned
multiply win is not attributable to the port: the same defect exists in the hand-written kernels and
is fixed for `develop` separately in vortex-data/vortex#9210, stacked on vortex-data/vortex#9211.
Re-measure the port against `develop` once that lands, because the comparison above flatters it.

### Measured dead ends

Recorded so they are not retried. All of these are in vortex-data/vortex#9130 as well.

- Bounds-check elimination in the row loop is not available. Narrowing the varying view to the row
  count buys nothing, and `get_unchecked` is not uniformly a win: about 10% on `mul_u16` and
  `mul_u32`, and 22% slower on `mul_u8`.
- A per-argument row source that keeps the `Varying` view when another argument is batch-constant is
  4x slower than the `ArgColumn` branch it replaces, which already vectorizes.
- A batch-constant operand therefore still demotes its neighbours off the slice path. Closing that
  needs the row loop monomorphized over which arguments are constant. Revisit when `Compare` moves
  onto `RowFn`, since `col < literal` is exactly this shape.

`mul_i32_constant` is the one regression that survives, and it is inside this host's drift. Let
CodSpeed settle whether it is real.

### What this implies for #9129 and #9130

- The `RowFn` API needed nothing. No new visit method, no options-aware `sink_dtype`, no return
  witness. `NumericBinary::FALLIBLE = true` is conservative for every dispatch arm, and each
  concrete result type supplies the precise loop behavior.
- `SinkResult::Accumulated` and its two constraints belong in #9130, and are recorded there.
- On kernels this close to the vectorizer's decision boundary, the emitted IR is the reliable gate
  and wall clock on one host is not. Two separate interventions here moved a benchmark the wrong
  way, and host drift between sessions exceeded the effects under measurement.

### Final API cleanup and generated code

The later simplification did not add numeric-specific surface:

- `NumericBinary` declares `ARG_NAMES = &["lhs", "rhs"]` instead of repeating an argument witness.
- Its `Options = NumericOperator` has no persistence bound or implementation. The registered
  `Binary` function remains the sole owner of the serialized `vortex.binary` contract.
- The selected input tuple carries arity, dense-safety, decode fallibility, and filtered-decode
  cost. The selected sink and `SinkResult` carry output and deferred-error facts.
- `SinkResult` is sealed, but a numeric function does not need to implement it. It chooses the
  supplied unsigned evidence width that matches the primitive element width.
- `OutputSink` remains one abstraction. A later numeric function with multiple logical outputs
  should put both builders in one sink rather than add a pair-of-sinks framework type.

The final cleanup was checked against its parent by cross-compiling the optimized
`row_fn_executor` benchmark for `x86_64-apple-darwin` with `target-cpu=x86-64-v3`. After normalizing
symbol names and metadata, the vector/reduction block for checked `i64` add matched exactly. It
retains `<4 x i64>` loads and adds, vector overflow detection through xor/and/compare operations,
`<4 x i1>` OR accumulation, and a reduction after the loop. The vector body has no call or panic
path, and the scalar tail is unchanged.

The ordinary `ElementSink` and custom-sink wrapping-add monomorphs also matched exactly. Native
Apple M4 Max measurements over 65,536 rows found RowFn median changes between 1.11% faster and 0.94%
slower, with fastest changes within about 0.17%. Specialized controls drifted more than the RowFn
arms, so there is no measurable native regression from the cleanup.

This is not an x86 runtime result. It proves that the API edits preserved the optimized x86_64-v3
loop shape. Runtime confirmation for numeric changes should use the stable public benchmark names
from #9136 on the target host.

The next session will run on x86 and must perform that confirmation. The #9136 `binary_ops`
benchmark is on `develop` at `9a482c0230`, so compare this branch with the latest
`origin/develop` using the same public benchmark names. Record both exact commits and run each
revision at least twice in alternating order. If possible, pin one core. Report fastest and median
values with the CPU and timer configuration. If a stable case regresses, compare its optimized LLVM
IR before changing the row API or restoring hand-written execution.

### Verification

The whole of `binary/numeric/tests.rs` passed unchanged, including
`test_decimal_overflow_on_null_lane_ignored` and the integer-error tests that pin the valid-row
retry. Decimal is untouched and stays on `execute_numeric_decimal`. The final API state also
recorded 67 focused RowFn tests, 179 tensor tests, 230 geo tests, nightly formatting, and full
workspace clippy. Clippy needed `PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1` because the host
Python is 3.9 while the workspace targets the Python 3.11 stable ABI.
