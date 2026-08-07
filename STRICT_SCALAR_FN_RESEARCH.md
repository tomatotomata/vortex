<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# A layered authoring API for strict scalar functions

**Status: historical design record, with the final API review recorded at the end.** This document
keeps the experiments in the order they happened, including APIs and ports that were later removed.
The current architecture is one `RowFn` authoring trait, private lifting, one sink-backed
`RowVisitor::visit_prepared_into` primitive, and a deliberately open input/output vocabulary. Read
[`SCALAR_FN_HANDOFF.md`](SCALAR_FN_HANDOFF.md) for orientation, then the final section here before
using an earlier sketch.

> **Later architecture decisions:** `StrictScalarFnVTable`, the columnar ports, returning visits,
> both witness types, `PersistableOptions`, the public `NullHandling`, the aggregate decode-shrinks
> flag, and the unused `TensorSink` were deleted. Framework-only visitor, tuple, and result traits
> are sealed. `InputElement`, `OutputElement`, and `OutputSink` remain open so functions can add
> their own decode and output primitives. Sections below remain the evidence that led to those
> decisions, not the API to implement.

---

## Current benchmark and codegen record

The authoritative current comparison is the
[x86 AVX-512 re-measurement on issue #9128](https://github.com/vortex-data/vortex/issues/9128#issuecomment-5151831802).
It records the machine, exact refs, stabilized governor, two-run fastest and median results, control
limitations, geo fix, adaptive-null diagnostics, and native LLVM IR/assembly in folded sections.
It supersedes every older shared-VM or pre-#9076 figure in these notes for claims about the current
branch versus `develop`.

The run used candidate `d293d3cdd59e` plus the recorded geo bbox widening, baseline
`876996fe7846`, and a Ryzen 9 7950X pinned to CPU 4 with the TSC timer and performance governor.
The conclusions that survive into the implementation plan are:

- sink-only checked add is 1.018-1.226x faster by median than its benchmark-local specialized
  control, depending on constant and null shape;
- cosine is 1.40-30.13x faster than current develop;
- prepared overlapping `contains` is 8.60-8.77x faster by median, while the widened bbox gate
  restores disjoint polygons to parity with #9076;
- point/constant geo still has real 8.6-14.2% and 10.9-13.2% median regressions;
- `BytesLen` is 1.410-1.411x faster by median on long strings and 1.097x on short strings;
- the global 75% survivor threshold mispredicts both one-input/50%-null and
  two-input/10%-null geo cases, so adaptive selection needs element/arity-aware cost data;
- checked-add codegen has AVX-512 error-word accumulation and post-loop vector reduction with no
  per-row error branch; current `l2_norm` remains a strict-order scalar reduction.

The stabilized cosine median ratios preserve the shape and width dependence instead of collapsing
the result into one headline range:

| shape | width 2 | width 32 | width 256 |
| --- | ---: | ---: | ---: |
| column x column | 5.77-5.79x | 1.93-2.19x | 2.54-2.58x |
| column x constant | 12.44-12.48x | 28.87-28.96x | 30.08-30.13x |
| column x extension constant | 3.05x | 1.40-1.41x | 1.77-1.78x |

The final geo median ratios, including the bbox patch, are:

| predicate and shape | develop / row branch |
| --- | ---: |
| contains, column x column points | 0.951-0.964x |
| contains, column x column polygons | 0.999-1.003x |
| contains, constant x points | 0.876-0.921x |
| contains, disjoint polygons | 0.993-0.997x |
| contains, overlapping polygons | 8.60-8.77x |
| intersects, column x column polygons | 0.983-0.995x |
| intersects, points x constant | 0.884-0.902x |
| intersects, disjoint polygons | 1.011-1.019x |
| intersects, overlapping polygons | 1.011-1.023x |

Here, as above, ratios greater than 1x favor the row branch. The issue comment contains the paired
fastest and median observations rather than only these compact ranges.

The historical measurements below remain because they explain design decisions and experiments made
while building the prototype; they are not the current before/after performance record.

---

## The design in one screen

```text
RowFn ──────────blanket──▶ StrictScalarFnVTable ──────blanket──▶ ScalarFnVTable
(row at a time, types                (null / constant / validity          (full control)
 chosen per batch)                    lifting for a columnar kernel)
```

Two authoring traits, one for each axis a strict function actually varies on, plus a third axis (*how
a row is typed, and how its output is delivered*) factored into an open element and sink vocabulary that
neither trait mentions.

### `StrictScalarFnVTable`, the null/validity lifting

Write the structural metadata plus one **columnar** kernel that ignores validity. A blanket impl
derives:

- `is_strict = true`, and a mirrored `validity` a kernel can answer with the conjunction of its child
  validities when it never turns a wholly non-null row into a null (see
  [Strictness is not totality](#strictness-is-not-totality)), so the planner knows which rows are null
  without executing the function.
- `return_dtype` = `return_element_dtype` widened to nullable iff any input is nullable, so the
  strictness dtype contract holds by construction rather than per function.
- `execute` = the shared cases before the kernel runs: a null-constant input short-circuits to an
  all-null constant, all-constant inputs evaluate one row and broadcast, and partially-null inputs
  are handled per `NullHandling` (`Dense` masks after a full pass, `Filter` filters then scatters).
- Options serde, from `PersistableOptions` on the options type.

This is the layer for a function whose kernel is columnar rather than row-at-a-time: `not` (one `!`
per 64-bit word), `list_length` (a difference of offset buffers), `list_sum` (a grouped accumulator over
the elements child). See [Why three concepts and not fewer](#why-three-concepts-and-not-fewer) for why it
cannot be folded away.

### `RowFn`, one row with element types chosen per batch

Name a witness argument tuple and return type, then in `dispatch` pick the concrete element types for
a batch and hand the framework a row closure through a rank-2 visitor. A blanket impl derives the
whole `StrictScalarFnVTable` from it. When the element types are fixed, `dispatch` is a single
`visit` at those types. When one ID spans several widths (`l2_norm` accepts f16/f32/f64), `dispatch`
matches on the input dtypes and visits at the chosen width.

Everything structural follows from the argument tuple and return type: arity, per-argument dtype
validation, the output dtype, null handling, and fallibility. There is nothing for an implementor to
declare twice or get wrong, because the framework reads it off the types (see
[Properties, not conventions](#properties-not-conventions)). A constant operand is decoded once and
read at stride 0, so a broadcast argument costs one decode rather than one per row.

Output takes one of two forms, chosen per visit. `visit` takes a closure that **returns** an
`OutputElement`, one owned value per row whose dtype is a property of its Rust type. `visit_into` takes
one that **writes** into an `OutputSink`, allocated once per batch knowing the output dtype and handing
out a place to write. Orthogonally, `visit_prepared` runs a once-per-batch prepare step over the
element values of whichever operands are constant for the batch, and threads its result to every row
by shared reference; plain `visit` is that with unit state (see
[Constant compute](#constant-compute-the-last-quadrant-of-the-lifting)). The sink carries what an owned per-row value cannot: `l2_denorm` writes each row
into a slice of one flat buffer, so its output width comes from the arguments and it allocates once
rather than per row. The executor holds the sink and passes the handle in, so the closure stays `Fn`
and the returning path pays nothing.

Note that `RowFn` does not *require* totality, it just cannot currently express its absence: both output
forms build an all-valid column, so a row kernel has no way to say "this row is null". An
`impl OutputElement for Option<T>`, or a sink that can push a null, would lift that, at the cost of
revisiting the `validity` law that reads the output validity off the inputs. No function needs it yet, so
it is not there.

### The element vocabulary, how a row is typed

`InputElement`, `OutputElement` and `OutputSink` are open traits. A `NativePType`, `bool`, `Bytes` (a
resolved `&[u8]`), and `BytesLen` (a length read from a view without resolving it) ship in the framework,
and `vortex-tensor` adds `TensorRow<T>`, reaching through the extension wrapper into flat storage, plus
`TensorSink<T>` on the output side, in its own crate. Adding `&str`, decimals, or a list row is one impl
that every row function gains, with no framework change.

---

## Why three concepts and not fewer

The standard applied here: every trait, and every member of every trait, has to have a purpose
nothing else can provide. Testing each against that standard is what the bulk of this research was.

### `RowFn` and the witnesses are forced, not chosen

A scalar function's *signature*, meaning its arity and fallibility, is a property of
`(function, options)` with **no input dtypes**: `ScalarFnVTable::arity(&self, options)` and
`is_fallible(&self, options)`, and `ScalarFnSignature` above them, take none. So any framework that
derives arity and fallibility from element types has to be able to name element types *without seeing
dtypes*, which is exactly what `ArgsWitness` / `RetWitness` are. Because `dispatch` *does* see dtypes
and could choose otherwise, some check has to tie the two together, which is the compile-time witness
check below. This cost is not a consequence of the rank-2 encoding: **any** design that derives a
dtype-free signature from per-batch types pays it.

A previous iteration made the width choice a generic-associated-type family generated by a
`row_family!` macro. Rust cannot abstract over a GAT's bound (`type Args<T: Self::Bound>` is
rejected), so that approach needed a trait *and* an adapter per width class, hand-written or
macro-stamped. The rank-2 visitor sidesteps the limit rather than writing around it: the kernel owns
the width `match`, where `T: Float` appears literally inside a `match_each_*_ptype!` arm, and the
framework method `RowVisitor::visit<A: ElementTuple, R: ApplyResult>` is generic only over bounds it
owns. The macro, its family traits, and its generated adapters are all deleted. Note that `dispatch`
is not even per-*width*: it can pick different element *kinds* per dtype, which no
bound-parameterized family could.

### `ElementwiseFn` was not forced, so it is gone

An earlier revision had a third trait, `ElementwiseFn`, for the fixed-element-type case: name `Args`
and `Ret`, write `apply`. It read cleanly, but it failed the standard. `RowFn` already covers the
fixed case (the dispatch is a single constant `visit`), so `ElementwiseFn` bought roughly seven lines
on exactly one production function (`byte_length`) at the cost of 114 framework lines and a third
link in the blanket-impl chain. The probes settled it: of the functions examined, `not` and `list_sum`
turned out not to be row functions at all, and `list_length` needed the encoding-aware
`reduce_encoded` hook that `ElementwiseFn` never exposed. So the constituency I expected it to have
never materialized, and it is deleted. `byte_length` writes a two-line `dispatch` instead.

The one-trait-with-defaults alternative (a single `RowFn` with `dispatch` defaulted to visit the
witnesses and `apply` defaulted to `unimplemented!()`) was rejected because it converts a compile
error into a runtime panic: a type implementing neither method compiles, registers, and answers
signature queries with a plausible shape, then panics on first execution. `dispatch` is therefore
required.

### `StrictScalarFnVTable` cannot be folded into `RowFn`

`RowFn`'s type surface is *closed*. The output dtype is `OutputElement::element_dtype()`, drawn from
the finite set of `OutputElement` impls, `ElementTuple` exists only for arities 1 to 3, and the loop
is one `apply` per row. Three whole classes of strict function are therefore inexpressible as a
`RowFn` at any cost:

- **Output dtype outside the element set.** `ext_storage`'s output is an extension array's storage
  dtype, so `vortex.geo.box` is a struct and `vortex.uuid` is a `FixedSizeList(u8,16)`. `vortex-geo`'s
  zone-map pruning calls `ext_storage` on a `geo.box` statistic, and a row-function port breaks it at
  plan time.
- **Variadic arity.** `merge` and `select` take an unbounded number of children, while `RowFn` fixes
  `Arity::Exact(n <= 3)`.
- **Sub-row-granular kernels.** `not` negates one 64-bit word at a time, so a row loop over `bool` is
  ~64x the memory traffic and, measured, 406x slower at a 64Ki batch (see
  [Measurements](#measurements)).

So the middle layer has a genuine, disjoint constituency: `not`, `list_length`, `list_sum`, and
prospectively `select`, `merge`, `json_to_variant`. "Just a visitor" collapses three concepts to two
rather than to one.

### Every remaining member earns its place

A member-by-member audit, with call sites found by grep rather than by guess, turned up nothing
deletable. The non-obvious cases are worth recording:

- **`RowVisitor::Out`** is what lets one `dispatch` `match` serve both plan time (`Out = DType`,
  validate and name the output dtype) and run time (`Out = ArrayRef`, decode and run the loop). The
  alternatives, a `{DType, ArrayRef}` enum unwrapped at each site or two separate dispatch hooks,
  either add unwrap-panics or duplicate the width `match` in every width-polymorphic function with no
  compiler check that the two copies agree.
- **A plan-time visit is unavoidable.** `l2_norm` declares `RetWitness = f64` but dispatches over
  f16/f32/f64, so the output dtype read off the witness would be wrong for two of three widths. Also
  `TensorRow<T>::validate` rejects an `f32` column against an `f64` witness, and the visit is what
  gives cross-argument uniformity for free (`int_max(i16_col, i64_col)` is rejected by
  `(T, T)::validate`, not by any `dispatch` body, which only inspects `args[0]`).
- **`ApplyResult` distinct from `OutputElement`** is what lets one trait serve both infallible
  (`Ret = f64`) and fallible (`Ret = VortexResult<f64>`) kernels without a wrapper. `f64` cannot be
  simultaneously fallible and infallible, so the fallibility bit lives on the return *shape* rather
  than on the element.

---

## Properties, not conventions

The framework's real value beyond line count is that two invariants an implementor used to have to
get right are now derived from the types, so an unsound combination cannot be written.

### Null handling follows from the arguments and the return type

`NullHandling::Dense` runs the kernel over every row including those behind nulls, then masks. It is
cheaper than filtering and the only option that leaves inputs at their original encoding, so it is
right whenever it is sound. Soundness needs two things, every argument readable behind a null row and
an infallible computation, and both are already visible in the types:

```rust
const fn row_null_handling<A: ElementTuple, R: ApplyResult>() -> NullHandling {
    if A::DENSE_SAFE && !row_is_fallible::<A, R>() { NullHandling::Dense } else { NullHandling::Filter }
}
```

Whether a dense read is safe is a property of the *element*, not of the function: reading a whole
value out of a flat buffer is safe (`NativePType`, `bool`, `TensorRow`, `BytesLen`), while following a
stored offset into a data buffer is not (`Bytes`), because arrays only validate the views of their
*valid* rows. This caught a real bug in this branch's own `byte_length`, see
[Problems to extract](#problems-to-extract-onto-develop).

### Fallibility comes from the return type *and* the element decode

A function is fallible if its computation can fail (`Ret = VortexResult<T>`) **or** if decoding an
argument can fail on legal data. The second source is real and was missing: `geo_distance`'s row
computation cannot fail, but parsing WKB bytes into a geometry can, for a *valid* row holding
malformed bytes. So `InputElement` carries `DECODE_FALLIBLE`, and fallibility is the disjunction:

```rust
const fn row_is_fallible<A: ElementTuple, R: ApplyResult>() -> bool { A::DECODE_FALLIBLE || R::FALLIBLE }
```

`is_fallible` gates dict-value pushdown (`arrays/dict/compute/rules.rs`), which speculatively
evaluates a function over *unreferenced* dictionary values, so a function that under-reports
fallibility fails a query on rows it never needed.

### The witness is checked at compile time

Arity, dense-safety and fallibility must not vary between the choices `dispatch` makes, because the
framework acts on them before dispatching. Since (with `ElementwiseFn` gone) *every* function names
its element tuple twice, once as `ArgsWitness` and once in the `visit`, the check that the two agree
is load-bearing, and it is a compile-time `const` assert inside each visit:

```rust
const fn assert_witness_agrees<F: RowFn, A: ElementTuple, R: ApplyResult>() {
    assert!(A::ARITY == <F::ArgsWitness as ElementTuple>::ARITY, "…");
    assert!(A::DENSE_SAFE == <F::ArgsWitness as ElementTuple>::DENSE_SAFE, "…");
    assert!(row_is_fallible::<A, R>() == row_is_fallible::<F::ArgsWitness, F::RetWitness>(), "…");
}
```

Monomorphizing any dispatch arm evaluates it, so even a `match` arm that never runs at a given width
is checked, and a disagreement fails the build pointing at the exact `visit::<…>` call. It compares
the raw arity/dense-safety/fallibility rather than the derived `NullHandling`, which collapses
dense-safety and fallibility together and would miss an arm that flipped both. A `compile_fail`
doctest pins that a lying witness does not compile. This replaced a runtime check that ran three
times per array (plan, execute, deserialize).

---

## Strictness is not totality

This is the finding that decides what the middle layer may derive. Note that
[#9033](https://github.com/vortex-data/vortex/pull/9033) reached the same conclusion independently and
has since landed, so this section is no longer the argument for the finding, only for the API that
follows from it.

Before #9033, the `is_strict` documentation stated the validity-equivariance law,
`f(…, mask(aⱼ, m), …) == mask(f(…, aⱼ, …), m)`, and then asserted as "consequence 1" that output
validity is the conjunction of input validities. **Consequence 1 does not follow from the law.** It
needs an extra premise: that the kernel never turns a wholly non-null row into a null. #9033 replaced
that equality with a one-sided bound, `valid(f(a₁, …, aₖ)) ⊆ valid(a₁) ∧ … ∧ valid(aₖ)`, which is the
vocabulary this branch uses. `docs/strictness-and-validity-pushdown.typ` proves the law and the
null-propagation reading are the same property, and separates what does not follow from either.

`list_sum` is the counterexample. Summing a valid *empty* list yields null. It still satisfies the law
(a null it introduces at a valid row appears identically on both sides of the equation and cancels),
so it is genuinely strict, but its output validity is *narrower* than its input validity.

Two properties, then, not one:

| property | what needs it |
| --- | --- |
| **strict** (null propagation, equivalently validity equivariance) | every validity push-down, the thing we actually want |
| **total** (non-null in implies non-null out) | upgrading the `⊆` bound to `=`, so validity is precomputable |

The old blanket impl derived `validity = union_child_validities` for *every* implementor, which needs
totality while the trait only requires strictness. Every current implementor happens to be total, so
nothing was broken, but a partial function joining the layer would get a `validity` that contradicts
what it computes: `arr.validity()` would report all-valid while `arr.execute()` yields the null, since
`ValidityVTable<ScalarFn>::validity` evaluates the derived expression. `list_sum` was about to be
exactly that, and is now ported onto the layer as the first non-total member.

#9033 says a function satisfying the stronger equality "can advertise that through
`ScalarFnVTable::validity`". That is the same idea as `is_total`, moved from a hand-written method to a
boolean, because a blanket impl cannot hand-write `validity` per function: it needs the property as
data in order to decide whether to derive one.

The fix needs no new property. `validity` is mirrored on `StrictScalarFnVTable` alongside `reduce`,
defaulting to `None`, and a kernel that satisfies the equality answers it with
`union_child_validities`. The unsound direction is the one that now takes work, and the safe default is
what a function gets for free.

An earlier revision of this branch instead added an `is_total` method and derived `validity` from it.
That was strictly worse: it introduced a concept the codebase did not have, in order to compute
something a function can just say directly. It is gone. The `RowFn` blanket impl answers `validity`
for every row function, justified by its own output vocabulary (no `OutputElement` is nullable, so no
row kernel can introduce a null), which keeps the row layer at zero boilerplate.

Note that strictness rather than totality gates membership either way: `is_null` is total but
disqualified, because it inspects validity and so does not propagate nulls. That is also why the trait
is not called `TotalFnVTable`.

> **A related latent issue, deliberately not fixed here.** Four functions declare `is_strict = true`
> and are strict-but-not-total: `get_item` (a nullable field under a non-null struct), `mask`,
> `variant_get`, `geo_envelope`. None is broken today, since `get_item` leaves `validity` at the
> default and `mask` overrides it correctly, but any that grows a conjunction-shaped `validity`
> derivation would be wrong. This predates the branch and belongs in its own investigation.

---

## Problems to extract onto develop

The framework surfaced three problems that are not really about the framework. Each is filed
separately and I think each should land as its own PR rather than riding in on this one. Note that
none of them is a live miscompute on `develop` today, which is worth saying plainly, because the
branch's own commit messages describe fixes to *this branch's* code.

1. **Strict-but-non-total validity derivation ([#9091]).** The `is_strict` documentation presents
   totality as a consequence of strictness when it is an independent premise (see above). Nothing
   derives validity from `is_strict` automatically, so nothing is wrong today, but the doc invites the
   next strict-but-partial function to write `validity: union_child_validities` and be silently wrong.
   **Superseded by [#9033], which lands the documentation correction on `develop`.** This branch needs
   nothing beyond that, since it now mirrors `validity` rather than deriving it from a property.

2. **Views behind null rows are unvalidated ([#9090]).** `VarBinViewArray::validate_views` only
   validates the views of *valid* rows, so a legal array can hold a view behind a null row naming a
   buffer that does not exist, and resolving it densely panics (`index out of bounds: the len is 1 but
   the index is 9`). On this branch, expressing byte length as "a function of the row's bytes" quietly
   changed *what gets decoded* and hit that panic. The fix here reads the length out of the view
   (`BytesLen`) and never resolves the row, and
   `test_byte_length_ignores_unresolvable_views_behind_nulls` pins it (verified to panic without the
   fix). `develop`'s `byte_length` was already immune, since it also read `view.len()`, so the
   extraction is that regression test rather than a code change. The doc half is also covered by
   [#9033], which deletes the dense-evaluation "consequence 2" outright rather than narrowing it. That
   leaves `InputElement::DENSE_SAFE` as the only place the licence is written down, per element rather
   than as a blanket claim, which is where it belongs.

3. **Bit-at-a-time bool packing ([#9092]).** `OutputElement for bool` used `BitBuffer::from_iter`,
   where the `Vec<bool>` is already owned and contiguous so `BitBuffer::from` routes to the
   multiversioned SIMD packer. Measured **6.6 to 7.9x faster** on the packing step, for every
   bool-returning row function. Note that `OutputElement` only exists on this branch, so the
   develop-side instance of the same pattern is a different call site:
   `encodings/sequence/src/compute/compare.rs` builds an n-bit result with a per-row predicate when it
   already knows the single set index. I have not benchmarked that site.

[#9033]: https://github.com/vortex-data/vortex/pull/9033
[#9090]: https://github.com/vortex-data/vortex/issues/9090
[#9091]: https://github.com/vortex-data/vortex/issues/9091
[#9092]: https://github.com/vortex-data/vortex/issues/9092

---

## Audit: can the four `StrictScalarFnVTable` impls really not be `RowFn`?

There were exactly four in production when this audit ran. Auditing each against the two questions that
matter, rather than repeating the earlier verdicts, **not one of them was structurally impossible**. Every
"cannot" in this document was really "cannot with the trait signed as it is today". One of the four,
`l2_denorm`, has since moved onto `RowFn`, so three remain. Recording the distinction because it is the
difference between a limit and a decision.

| function | signature expressible? | kernel row-shaped? | what it would take |
| --- | --- | --- | --- |
| `not` | **yes**, `(bool,) -> bool`, both elements exist | **no** | nothing. It can be a `RowFn` today and should not be: `!bits` is one `!` per 64-bit word, in place when unshared, against 16k closure calls and a `Vec<bool>` repack |
| `list_length` | output is a fixed `U64`; input needs a `ListLen` element | **no** | one new element. Still should not: the answer is a child array or one constant |
| `list_sum` | output is one number per row, so nearly: only the *nullability* is unexpressible | **no** | `impl OutputElement for Option<T>` and a list element, but the kernel is the real blocker |
| `l2_denorm` | **yes, now**: an `OutputSink` names its dtype from the arguments | yes, per-row scaling | **done**, see below |

**A varying output dtype was already supported, and listing it as a blocker was wrong.** `dispatch`
chooses element types per batch and `return_element_dtype` routes through it, so `R::Out::element_dtype()`
is already answered per dispatch arm. `l2_norm` relies on this today, visiting `::<(TensorRow<T>,), T>`
with `T` ranging over the float widths. The compile-time witness check pins only arity, dense-safety and
fallibility, deliberately leaving the output type free to vary. What `l2_denorm` needed was different and
narrower: its output dtype depends on the input *dtype* in a way no choice of element type can express,
because the extension dtype carries a shape. That is what `OutputSink::sink_dtype(args)` supplies.

**`list_sum`'s output side is the easy part; its kernel is not.** One number per row means it needs only
a nullable output element, no write-into-buffer machinery. But `execute_strict` is not a per-row sum: it
builds a `GroupedAccumulator` over `Sum`, calls `accumulate_list`, and then `mask_empty_lists` computes
per-group emptiness with `count_range` popcounts, with all-true and all-none fast paths and an early
return when nothing needs masking. Porting it to a row loop would hand-roll the shared aggregate
framework, lose the overflow modes that `NumericalAggregateOpts` selects, and trade SIMD popcounts for
per-row checks. That puts it in the same category as `not`: expressible, and worse.

So `l2_denorm` was the only one of the four whose kernel actually wants to be a row loop, which is why it
was the right first target despite needing the larger output-side change.

Two readings follow.

**The honest framing is "can, and here is whether it is worth it."** For `not` and `list_length` the
answer is a flat no on performance grounds, and those are settled. For `list_sum` the answer is
yes-with-changes, and the change it wants is a nullable output, which the sink could supply but which the
`validity` law argues against (see below).

**`l2_denorm` was the one worth doing, and it is done.** Its kernel genuinely is per-row scaling, and
it carried the `unsafe` the other three tensor ports removed. What it needed was a second visit method
whose closure *writes* its row instead of returning it, generalized to an `OutputSink` rather than
hardcoding `&mut [T]`, because the same mechanism covers three gaps recorded separately in these notes:

- **runtime-shaped output**: the sink is a preallocated flat buffer and the per-row handle a
  `&mut [T]` slice of it, so `l2_denorm` allocates once per batch rather than once per row. This is what
  shipped.
- **`str -> str` without the double copy**: the sink is one growing byte buffer plus views, and
  `upper`/`lower`/`replace` push into it. Strictly better than the `Cow` output element considered
  above, which still copies each row once. Not built, but the trait admits it unchanged.
- **nullable output**: a sink *could* push a null, which would remove the need for
  `impl OutputElement for Option<T>` as a separate patch. Deliberately **not** taken: both output forms
  build an all-valid column today, and that is exactly what lets the blanket `validity` return
  `union_child_validities`. Adding nulls to either form has to come with that law being revisited.

### What shipped

```rust
pub trait OutputSink: 'static + Sized {
    type Row<'a> where Self: 'a;
    fn sink_dtype(args: &[DType]) -> VortexResult<DType>;
    fn with_capacity(rows: usize, dtype: &DType) -> VortexResult<Self>;
    fn row(&mut self, index: usize) -> Self::Row<'_>;
    fn finish(self) -> VortexResult<ArrayRef>;
}

fn visit_into<A: ElementTuple, S: OutputSink, R: SinkResult>(
    self,
    apply: impl Fn(A::Elems<'_>, S::Row<'_>) -> R,
) -> VortexResult<Self::Out>;
```

**The executor threads the sink, not the closure**, so `apply` stays `Fn` and the existing `visit` pays
nothing. That was the design constraint, not an accident: relaxing `visit` itself to `FnMut` measured at
8 to 11% (see the `like` discussion), and a handle passed in per row avoids captured mutable state
entirely. Measured after the fact, `l2_norm` is unchanged at 69.05 µs against the 69.44 µs recorded
before the sink landed.

**Step 1 of the earlier plan turned out to be unnecessary.** The plan called for widening
`OutputElement::element_dtype()` to take `args`. It never happened, because `sink_dtype(args)` puts the
argument-dependence on the *sink* instead, leaving all three existing `OutputElement` impls untouched.
That is the better split: an element's dtype genuinely is a property of its Rust type, and only the
thing that needs the arguments asks for them.

**The `RetWitness` split resolved as predicted.** It carried two roles, *what dtype* and *is it
fallible*, and only the second is readable before `dispatch` picks a form. So `RowResult` now holds just
`const FALLIBLE`, with `ApplyResult: RowResult` adding the output element and `SinkResult: RowResult`
adding nothing but the error, and `RowFn::RetWitness` is bounded by `RowResult`. A returning dispatch
names `f64` or `VortexResult<f64>`; a writing one names `()` or `VortexResult<()>`. Coherence permits
this: `impl RowResult for ()` does not overlap `impl<T: OutputElement> RowResult for T` because
`(): OutputElement` does not hold and no downstream crate can make it hold, the same negative reasoning
the pre-existing `ApplyResult` impls already relied on.

**A new limit, worth naming.** `sink_dtype` sees the input dtypes but **not** the function's options,
because `OutputSink` does not know the `RowFn`'s `Options` type. A function whose output dtype depends
on an option value therefore still drops to `StrictScalarFnVTable`, whose `return_element_dtype` sees
both. Nothing in the repository needs it, and threading options through later is additive.

### Results

`unsafe` in `l2_denorm.rs` went from 8 blocks to 6. The two removed are the memory-safety ones on the
kernel path: `FixedSizeListArray::new_unchecked` in the constant-norms path, now `try_new` (the norm is
cast to the element dtype first, so the product stays non-nullable and the check passes), and
`PrimitiveArray::new_unchecked` in `build_tensor_array`, now `new`. That second one is an independent
cleanup rather than something the port forced.

The 6 remaining are not of that kind and are not the row layer's business: four are calls to
`L2Denorm::new_array_unchecked`, an `unsafe fn` whose contract is the *semantic* unit-norm invariant and
not memory safety, and two are buffer pushes inside `normalize_as_l2_denorm`, a helper that builds the
normalized child and is not a scalar function at all.

**Performance: the sink is faster than the kernel it replaced**, which was not the expected outcome.
`vortex-tensor/benches/l2_denorm.rs`, `fastest` column, both configurations run twice, 16384 rows,
non-nullable. The control implements `StrictScalarFnVTable` with the pre-port body, so it shares the
strict lifting and the gap is the row layer alone:

| width | sink | pre-port kernel | ratio |
| --- | --- | --- | --- |
| 2 | 88.02 / 88.16 µs | 60.19 / 60.45 µs | sink 1.46x slower *(since fixed, see below)* |
| 32 | 482.0 / 515.5 µs | 1.175 / 1.014 ms | sink **2.1x faster** |
| 256 | 10.23 / 10.43 ms | 20.41 / 22.48 ms | sink **2.0x faster** |

The likely cause of the win is that the pre-port kernel collected a `flat_map` over rows into a fresh
`Buffer<T>`, and `flat_map` is not `TrustedLen`, so that `collect` grew the buffer with a capacity check
per element. The sink allocates once with `BufferMut::zeroed` and each row writes a slice of it, which
vectorizes. The zeroing is not a separate pass at these sizes, since large allocations come back zeroed
from the allocator. This is a hypothesis consistent with the width scaling rather than something
profiled.

Width 2 showed the same regression as `l2_norm`'s, and for the same reason: both read tensor rows through
`TensorRow`, whose `get` re-derived a typed slice per row. Typing the column at decode time took
`l2_denorm` from 88.0 µs to **48.9 µs** at width 2, ahead of this control rather than behind it. See
[the like-for-like comparison](#the-like-for-like-comparison-and-the-per-row-cost-that-was-hiding-in-it)
for the measurement and for the wrong diagnosis it corrects.

The constant-norms fast path moved to `reduce_encoded`, which sees the argument arrays before the row
loop. It keeps both of its cases (unit norms return the normalized child untouched, any other constant
rewrites the storage elements through one multiply), and it still fires for a filtered batch because
filtering a constant yields a constant.

**Two visit methods do not cover everything, and it is worth being precise about the residue.** They
cover every function whose output is *computed* per row, returned or written. What stays columnar is
output that *aliases* its input, since `trim` and `substring` want to keep the input's data buffer and
rewrite only views, copying nothing, and a sink still copies bytes into itself. Likewise kernels whose
natural unit is not a row (`not`'s word-at-a-time negation, `binary`'s slice kernels) gain nothing.

The sink is also what a `str -> str` string library needs. After reclassifying `L2Denorm` as an
encoding, that string library becomes the prospective first production user rather than a second
one. The experiment still demonstrates that the generic sink can carry runtime-shaped and
builder-backed outputs without making the returning path pay, but it should not be stabilized from
the tensor experiment alone.

---

## Constant compute: the last quadrant of the lifting

The lifting's constant handling was complete on the data side and absent on the compute side. A
null-constant input short-circuits, all-constant inputs fold to one row, and a constant operand is
decoded once and read at stride 0. What nothing owned was kernel computation that depends only on a
constant argument: `cosine_similarity(rows, query)` with a broadcast query re-accumulated
`norm(query)`, an O(width) pass plus a sqrt, once per row, and the geo predicates rebuilt the
constant side's topology graph, R-tree, or bounding box once per row. `cosine_similarity` escaped
partially by hand-writing a `reduce_encoded` rewrite, and the survey found that rewrite already
wrong for the literal shape, which is the argument for framework ownership stated as a correctness
fact: one hand-written constant path per function is one place per function to rot on
encoding-normalization details.

### Where the hook can live, and where it cannot

The hoist needs three things at once: knowing which arguments are constant, having their decoded
values, and a typed place for the function to compute from them. Constness is a per-batch value
fact (a RunEnd slice landing inside one run, a per-chunk compression decision), so:

- **`dispatch` cannot see it.** It runs at plan time and run time and must choose identical element
  types at both; values do not exist at plan time.
- **Element types cannot encode it.** A `Const<E>` wrapper element would need value-aware dispatch
  to be chosen, splitting plan/run monomorphizations in exactly the way the witness deliberately
  does not pin, and costing 2^arity dispatch arms. The salvageable half of the idea,
  framework-internal value-driven specialization, already exists as the stride-0 `ArgColumn`.
- **The closure cannot memoize it.** An `unsync::OnceCell` capture compiles under `Fn`, but without
  constness information it is wrong (it would cache row 0 of a varying operand), and with that
  information it saves nothing over a prepare step while planting an unhoistable load inside the
  loop.

That leaves one point: inside the visit, after decode, where `ArgColumn` already knows each
column's stride. `ElementTuple` gains `ConstElems<'a>`, the element tuple with every slot wrapped
in `Option` (`Some` iff that operand is batch-constant), and the visitor gains:

```rust
fn visit_prepared<A: ElementTuple, P, R: ApplyResult>(
    self,
    prepare: impl FnOnce(A::ConstElems<'_>) -> P,
    apply: impl Fn(&P, A::Elems<'_>) -> R,
) -> VortexResult<Self::Out>;
```

`prepare` runs once per batch; its result reaches every row by `&P`, so `apply` stays `Fn` and the
loop keeps the shape the FnMut measurement forbids changing. `P` names no column lifetime, so
prepared state provably cannot alias the columns the loop reads. Plain `visit` is now a *provided*
method, `visit_prepared` with unit state: the ZST erases under monomorphization (measured, l2_norm
non_nullable at 33.38 us against the 32.83 us hand-written control, parity), the duplicate row loop
is deleted, and the visitor's method count grows with genuine axes (how output is delivered) rather
than with feature combinations.

`prepare` is infallible in v1: it refines values the row loop could compute itself, and fallibility
is read off the witnesses before dispatch, so a failing prepare would have nowhere to be declared.
The extension (prepare returning `VortexResult<P>`, riding the existing fallibility axis) is
documented next to the method and deliberately unbuilt, because no adopter needs it.

Three boundary facts worth stating because they will bite someone:

- **Prepare must never be load-bearing for validation.** An empty batch decodes every operand as
  non-constant (there is no row 0 to slice), so a prepare that validated its constant would
  silently not run. Validation belongs to `validate` and the dtype rules.
- **What counts as a batch constant is wider than the constant encoding.** The stride-0 decode sees
  one level through two wrappers that spell "the same value in every row" without being it:
  `MaskedArray(ConstantArray)`, how the compressor spells an all-same-with-nulls chunk (sound
  because the lifting owns validity entirely, so the value the loop reads behind a null row is
  unobservable), and `Extension` over constant storage, the shape extension builders produce before
  `ExtensionConstantRule` normalizes it.
- **`P` having no `Send`/`Sync` bound is load-bearing.** geo's `PreparedGeometry` carries
  `Rc`/`RefCell` and could not be prepared state otherwise. The flip side, recorded so it is a
  decision rather than a surprise: adding such bounds later (a parallel row loop, say) is a
  breaking change to real adopters, not a relaxation.

### What it bought, measured

**cosine_similarity, and a lesson in ILP.** The closure accumulated the rhs norm per element and
sqrt'd it per row, a third of the arithmetic plus one of two sqrts. Hoisting it moved the benchmark
by only ~5% at width 32 and ~3% at 256 (16384 rows, fastest column), far under the flop count,
because the loop is latency-bound on the serial dot-product FMA chain (FP reassociation is illegal)
and the removed accumulation was executing in the chain's spare ILP slots. The measurable saving is
the hoisted sqrt. The row is bit-identical either way, each arm accumulating in the same order as
the unprepared kernel.

The lesson generalizes and is the honest scoping of the feature: **"removes an O(width) pass per
row" is not "saves time" when that pass rides in ILP slack.** The work that collects the full
saving is work that extends the dependency chain: parses, tree builds, prepared structures. Which
is exactly what the geo numbers then showed.

**The geo predicates, where the win lives.** `contains` substitutes an owned
`PreparedGeometry<'static>` of the constant operand (r-tree plus self-noded topology, built lazily
inside `P` through a `OnceCell` so point-row batches never pay for it) into relate exactly where
geo routes `Contains` through relate, argument order preserved including the `MultiPolygon`
reversal; direct pairings keep geo's own algorithms untouched. `intersects` hoists the constant
side's `bounding_rect` and replays geo's own disjoint-bboxes early-out, gated to fire only where
geo makes exactly that comparison first. `distance` was investigated and left alone: geo builds
R-trees for both sides inside a private helper on every call, so there is no seam to reuse one, and
the finding is recorded as a doc comment on its dispatch. 16384 rows, fastest column, two runs:

| arm | before | after | change |
| --- | --- | --- | --- |
| contains, constant x polygons, overlapping | 457.5 / 458.0 ms | 50.88 / 50.00 ms | **9.1x** |
| contains, constant x polygons, disjoint | 7.04 / 7.05 ms | 3.97 / 3.74 ms | **1.9x** |
| contains, constant x points (direct route) | 3.15 / 3.08 ms | 3.22 / 3.15 ms | unchanged |
| contains, column x column | 3.56 / 6.29 ms | 3.68 / 6.40 ms | unchanged |
| intersects, polygons disjoint x constant | 6.81 / 6.72 ms | 3.20 / 3.14 ms | **2.1x** |
| intersects, polygons overlapping x constant | 9.57 / 9.48 ms | 9.87 / 9.63 ms | 1-3% slower, accepted |
| intersects, points and column x column arms | 3.20 / 5.98 ms | 3.21 / 5.92 ms | unchanged |

The overlapping-intersects arm is the disclosed tradeoff: the hoisted bbox check is an early-out,
so where it rarely fires the row pays for it. The port was an out-of-sample test of the API and
passed it: **zero framework changes were needed**, matching the element vocabulary's earlier record
(`TensorRow`, `GeometryRow`, `TensorSink`, each added in its own crate).

**Deleting the hand-written path made its shape faster.** With `Extension`-over-constant visible to
the stride-0 decode, cosine's `reduce_encoded` constant routing (manufacture an `L2Denorm` from a
constant operand, answer through the denorm paths) became deletable. Its shape then sped up:

| width | through the deleted rewrite | through the row loop + prepare |
| --- | --- | --- |
| 2 | 118.8 us | **63.08 us** |
| 32 | 554.0 us | **377.9 us** |
| 256 | 5.159 ms | **3.007 ms** |

Both constant spellings now measure identically (63.08 vs 62.72 us at width 2). The hand-written
fast path was 1.5-1.9x slower than the framework path that replaced it, on top of having missed the
literal shape entirely. That is the dedup argument in its strongest form: not fewer lines, but
fewer wrong ones.

### The one unenforceable thing

The design's benefit rests on LLVM treating the per-row branch on the prepared `Option` as
loop-invariant. Three outcomes exist per call site: unswitched (intended), if-converted (both arms
computed, the hoist silently evaporates while staying correct), or retained (a branch in a cheap
scalar kernel can block vectorization). For every real adopter the hoisted work is a loop or a
parse, which cannot be speculated, so the worst case degrades to one predicted branch per row, the
same cost class as the bounds check kept over `unsafe`. It is still a hope rather than a contract,
and the convention that polices it is stated in the trait-choice guide: every adopter lands with a
constant/non-constant benchmark pair, and the non-constant arm must not move.

### Rejected alongside

- **`Const<E>` wrapper elements**: needs value-aware dispatch; splits plan/run; 2^arity dispatch
  arms. Dead on the purity invariant.
- **Closure-internal `OnceCell` memoization**: wrong without constness plumbing, redundant with it.
  Distinct from the `OnceCell` *inside `P`* that contains uses, which is constness-aware and only
  defers an expensive build.
- **Plan-time currying through `reduce`** (folding a Literal into Options as a compiled variant):
  the only design that amortizes across batches, deferred because `PersistableOptions` admits only
  the source value, it misses every run-time-only constant, and re-currying bifurcates function
  identity, silently detaching encoding kernels keyed on the original function. Revisit only if
  per-batch prepare cost ever measures as material.
- **`visit_prepared_into`** (sink plus prepare): no user. `l2_denorm`'s constant case is a bulk
  answer in `reduce_encoded`, not a prepared loop. The asymmetry is deliberate and cheap to fix
  when a user appears.

---

## Is there anything left to port?

Asked directly: could the remaining hand-written vtables move onto `RowFn` if the element vocabulary
covered more types? Classifying all ~30 of them says no, and says the vocabulary is not what is
stopping them.

| blocker | count | members |
| --- | --- | --- |
| **Not strict.** `RowFn` implies strict, so these cannot reach it at all. | 12 | `between`, `case_when`, `cast`, `dynamic`, `fill_null`, `is_null`, `is_not_null`, `list_contains`, `pack`, `stat`, `row_size`, `zip` |
| **The answer already exists in bulk.** Zero-copy child projection, a metadata field, or a vectorized slice kernel. A row loop would be strictly slower. | 12 | `not`, `list_length`, `binary`, `mask`, `ext_storage`, `get_item`, `select`, `merge`, `variant_get`, `geo.envelope`, `json_to_variant`, `row_encode` |
| **No element rows to read.** Zero-arity, or a type-erasure adapter. | 5 | `literal`, `root`, `row_idx`, `row_count`, `ForeignScalarFnVTable` |
| **Output side.** Nullable output, or an output dtype that depends on runtime data. | 2 | `list_sum`, `geo.envelope` |
| **Value-dependent per-batch setup.** | 1 | `like` |

`geo.envelope` is the one function counted twice: its output is a struct-of-four extension type *and*
its fast paths hand back existing child arrays untouched.

`binary` deserves a note, since on strictness alone it looks portable: only its Kleene `And`/`Or` are
non-strict, and `is_strict` already varies by operator, so comparison and arithmetic go through the
strict lifting today. What keeps it columnar is the kernel. `collect_zip_bits` and `LaneZip` run over
`as_slice()` pairs as tight vectorizable loops, with a separate constant-operand path
(`collect_bits(lhs, |a| a.is_eq(rhs))`). Routing that through a per-row closure and `ArgColumn::get`
would give up the slice-level vectorization for nothing.

Three things follow.

**The porting well is dry.** The eight functions on `RowFn` (`byte_length`, the four tensor kernels, the
three geo kernels) are the complete set in this repository that wants a row loop. Every remaining one is
blocked, and forcing any of them onto `RowFn` would cost performance rather than save lines. `l2_denorm`
was the last one the vocabulary was actually keeping out, and the sink let it in.

**Missing elements are not the constraint.** Only `list_contains` would need new input vocabulary, and
it is independently blocked by non-strictness, so a list element would not unblock a single function
today. A list *input* element is nonetheless easy (`Bytes` already proves the shape: `Elem<'a>` is a
GAT, so `&'a [T]` works), and `list_length` could even be a `RowFn` given a `ListLen` element in the
style of `BytesLen`. It should not be, because its answer is a child array or one constant.

**`like` is a new gap, and the sharpest one.** It is strict, infallible, `(Utf8, Utf8) -> Bool`: on
signature alone it is the ideal `RowFn`. Two things block it, and measuring both is what settled where
it belongs.

Its constant-pattern path is fine. `reduce_encoded` already sees the argument arrays before the row
loop, so compiling the pattern once and evaluating in bulk has a home, and a constant operand stays
constant even through a filtered batch. No new hook needed for that case.

Its *per-row* pattern path is what blocks it. That path memoizes the compiled pattern across
consecutive rows carrying the same one, and a `RowFn` closure is `impl Fn`, so it can hold no such
state. Defeating the cache costs **5.7x** (`like_per_row_distinct_patterns` 249.1 µs against
`like_per_row_patterns` 44.03 µs, 2048 rows, same matching work in both), which is the same shape of
regression the constant-operand stride fixed for geo.

Relaxing the closure to `impl FnMut` would restore the cache, and it compiles as a one-word change.
It is not free. Measured on `byte_length_element`, `fastest` column, both configurations run twice:

| case | `Fn` | `FnMut` | delta |
| --- | --- | --- | --- |
| `long_strings_bytes_len` 4096 | 11.15 µs | 12.08 µs | +8.3% |
| `long_strings_bytes_len` 65536 | 166.4 µs | 181.7 µs | +9.2% |
| `long_strings_bytes_slice` 4096 | 14.75 µs | 15.97 µs | +8.3% |
| `short_strings_bytes_len` 65536 | 166.2 µs | 180.4 µs | +8.5% |
| `short_strings_bytes_slice` 65536 | 180.9 µs | 200.3 µs | +10.7% |

Capturing the closure by `&mut` inhibits the vectorization the shared capture allows, so `FnMut`
taxes every row function 8 to 11% to enable state that one function wants. Keep `visit` on `Fn`.

The conclusion is that `like` does not want a row loop at all: its general path needs cross-row state,
and its fast path is bulk. What it wants is to declare `(Utf8, Utf8) -> Bool` through the element
vocabulary and keep its own kernel, which is the missing cell below. A per-batch setup hook would not
have been enough on its own, since the state `like` needs is mutable *across* rows rather than fixed
before them.

A second, smaller thing blocks `like` too: it renders custom SQL through `fmt_sql`, and neither
`StrictScalarFnVTable` nor `RowFn` forwards that, so today porting any function with bespoke SQL
rendering would silently lose it.

---

## Known gaps and future work

Found by the porting probes, left unfixed here because each is a larger change with its own review
surface. Recorded so they are decisions rather than surprises.

- **~~No constant-operand affordance.~~ Fixed twice over.** A partially-constant call used to decode
  the constant column in full, so a broadcast operand cost one decode per row (measured: a broadcast
  query vector cost the same as a genuine column, 234 ms vs 226 ms at 50k x 256). That was what kept
  the geo functions off `RowFn`. Each decoded column now carries a stride, 0 for a constant, and the
  geo functions are row functions. Constant *compute* was the remaining half, closed by
  `visit_prepared` (see [Constant compute](#constant-compute-the-last-quadrant-of-the-lifting)).
- **`NullHandling::Dense` is chosen on safety alone, with no cost input.** For a fixed-width element
  (`TensorRow`) dense is unambiguously cheaper. For an unbounded-width row (a nested list) the garbage
  behind a null row need only be *in bounds*, so it can span the whole elements array, which is
  pathologically O(nulls x elements). No current function hits this, but the choice should consider
  width.
- **`OutputElement::build(Vec<Self>)` forces materialization.** A row function's output is always a
  freshly built `Vec` turned into a `PrimitiveArray`, so it cannot return a `ConstantArray` or a lazy
  child. This is why `list_length` is a columnar `StrictScalarFnVTable` rather than a `RowFn`, since a
  row port would materialize one `u64` per row and lose the `FixedSizeList` constant. A columnar output
  escape that stays inside the framework ("given the decoded columns, can you produce the whole output
  at once?") would let `list_length`, `byte_length` and `not` share one abstraction.
- **The missing cell.** The two authoring traits cover *declare-signature-once + row-loop* (`RowFn`)
  and *hand-write-signature + own-kernel* (`StrictScalarFnVTable`). The cell for
  *declare-signature-once + own-kernel* is empty, so a columnar function hand-writes five signature
  methods (`arity`, `child_name`, `return_element_dtype`, `null_handling`, `is_fallible`) that are all
  mechanically derivable from an element tuple.

  **It is buildable.** The obvious worry is coherence, since `RowFn` already blanket-impls
  `StrictScalarFnVTable` and a second blanket impl of the same trait is a hard E0119 conflict. The way
  through is to layer rather than branch, putting the new trait *between* the two:

  ```text
  StrictScalarFnVTable  <-blanket-  StrictSignature  <-blanket-  RowFn
  ```

  One blanket impl per edge, so nothing overlaps, and a columnar function hand-writes `StrictSignature`
  while a row function reaches it through `RowFn`. Compiling the shape confirms a hand-written impl
  coexists with the blanket one, including from a *downstream* crate, because within the crate that owns
  the type rustc can see the blanket impl's bound does not hold. This is not a new trick here:
  `impl<V: StrictScalarFnVTable> ScalarFnVTable for V` already coexists with `Like`'s and `Between`'s
  hand-written `ScalarFnVTable` impls the same way.

  **The user count is 3, not 12, and 2 of those need an element first.** Being in the columnar category
  is not enough: the function's *signature* has to be expressible in the vocabulary, and
  `element_dtype()` taking no arguments rules out every function whose return dtype is derived from its
  input at runtime. That is most of them: `mask` returns `arg_dtypes[0].as_nullable()`, `ext_storage`
  returns `ext_dtype.storage_dtype()`, `get_item` and `select` a projection of the input struct,
  `variant_get` an options-derived dtype, `binary` a width negotiated between operands. What is left is
  `not` (`(bool,) -> bool`, usable today), `like` (`(Bytes, Bytes) -> bool`, usable today once `fmt_sql`
  forwards), and `list_length` (needs a `ListLen` element in the style of `BytesLen`).

  So this is worth building *after* the elements that give it a third user, not before. Against ~140
  lines of new trait and blanket impl it would save roughly 20 lines per function, which at one usable
  caller is a wrapper with one impl. The cheap interim is to make `validate_row_args`,
  `row_null_handling` and `row_is_fallible` public, which turns each hand-written signature method into
  a one-liner and removes the *logic* duplication (each function currently rolling its own dtype check
  and asserting rather than deriving its null handling) without adding a layer.
- **No nullable output element, so no non-total `RowFn`.** `OutputElement::build` always produces an
  all-valid column, so a row kernel cannot return a null from a valid row. `impl OutputElement for
  Option<T>` is the whole fix. Left out because nothing needs it *yet*: `list_sum` would need it, but
  is columnar for independent reasons too (the grouped-accumulator path and the `FixedSizeList`
  constant).
- **No borrowed output element, so no zero-copy row function.** A row closure returns an
  `ApplyResult`, which is `'static`, so its result cannot borrow from the input columns. Note the
  asymmetry with the input side, where `InputElement::Elem<'a>` is a GAT and borrows freely. Every
  `str -> str` function therefore copies: `OutputElement for String` allocates one `String` per row
  and then rebuilds views from them. A string library would hit this on its first `upper`. Two
  distinct fixes, of increasing scope:
  - `upper`, `lower` and `replace` genuinely allocate, and want a `Cow<'a, str>` output element. That
    needs `OutputElement` to grow its own lifetime GAT and `build` to take an iterator rather than a
    `Vec`, so a borrowed row passes through without a copy and an owned one is built in place.
  - `trim`, `substring`, `left` and `right` want more than a `Cow` can give. Their result is a
    *slice* of the input, so the right kernel keeps the input's data buffer entirely and rewrites
    only the views, copying no bytes. That stays columnar whatever the output element can express.

  Predicates and measurements (`starts_with`, `contains`, `byte_length`) have none of this problem
  and are already the best case for `RowFn`, so the split for a string library falls along the return
  type rather than the argument type.

  **A plain higher-ranked bound does not get there,** which is worth recording because it looks like
  it should. Writing the visit as `impl for<'a> Fn(A::Elems<'a>) -> R::Elem<'a>` fails with
  [E0582]: the `Fn` sugar puts `R::Elem<'a>` in an `Output` binding, and rustc requires the bound
  lifetime to appear *structurally* in the trait's input types before a binding may reference it. An
  opaque projection `A::Elems<'a>` does not count, even though it plainly mentions `'a`. Three routes
  around it, measured by compiling each:

  | route | works | cost |
  | --- | --- | --- |
  | concrete input type instead of `A::Elems<'a>` | yes | gives up the element abstraction |
  | custom callable trait with a generic `apply` method | yes | callers write a struct per kernel, not a closure, and the impl must spell `<Bytes as InputElement>::Elem<'a>` rather than `&'a str`, or hit [E0195] |
  | pass a zero-sized `Row<'a>(PhantomData<&'a ()>)` token beside the row | yes | closures survive, but every row closure grows an ignored parameter |

  The third is the one to build on: the token makes `'a` appear structurally in the `Fn`'s inputs,
  which satisfies E0582 and lets the `Output` binding reference it, and plain closures still infer.
  The ignored parameter is a tax on *every* row function though, so the shape to prefer is a second
  visit method for lending kernels, leaving today's `visit` untouched for the `'static` majority.

  **Still open, and not what `visit_into` is.** The sink method added since is a second visit method, but
  for a closure that *writes* rather than one that *lends*: its output is owned by the sink, not borrowed
  from the row. A lending visit would still need the `Row<'a>` token. The precedent it sets is that
  adding a third visit method costs the existing ones nothing, which is the same additive shape.

  [E0582]: https://doc.rust-lang.org/error_codes/E0582.html
  [E0195]: https://doc.rust-lang.org/error_codes/E0195.html
- **~~`OutputElement::element_dtype()` takes no arguments,~~ Resolved, and not the way this predicted.**
  An element's output dtype is a property of its Rust type and cannot depend on runtime data, which is
  what kept `l2_denorm` columnar: it returns whole tensor rows, and a tensor's dtype carries its shape.

  Calling that a law was wrong, and the fix was recorded here as "widen `element_dtype` to take `args`".
  That is *not* what shipped, and the shipped version is better. `OutputSink::sink_dtype(args)` puts the
  argument-dependence on the sink, so all three `OutputElement` impls keep their no-argument
  `element_dtype()` and only the thing that needs the arguments asks for them.

  This gap also named the real blocker correctly: `build(values: Vec<Self>)` with `Self = Vec<T>` means
  one heap allocation per row and then a flatten, against a columnar kernel that scales the flat storage
  buffer in a single pass. At 16k rows that is 16k allocations versus zero, and no amount of dtype
  plumbing fixes it. The prescription it drew, "an output element that writes into a preallocated flat
  buffer (`fn apply(row, out: &mut [T])`)", is exactly what `OutputSink` is, generalized past `&mut [T]`
  so a byte buffer works too. See
  [the audit](#audit-can-the-four-strictscalarfnvtable-impls-really-not-be-rowfn) for what it cost and
  bought.

  Note also what *not* to do on the input side: replacing the generic `TensorRow<T>` with a
  non-generic element whose `Elem<'a>` is an enum over `f16`/`f32`/`f64` would move the width choice
  from monomorphization into a branch inside the row loop. That is precisely what
  `match_each_float_ptype!` plus a generic element exists to avoid, so it would cost every tensor
  kernel its inner-loop specialization.
- **~~The witness carries four scalars through two associated types.~~ Not a gap.** This looked like
  the framework's weakest joint, since `ArgsWitness` and `RetWitness` are read *only* for `ARITY`,
  `DENSE_SAFE`, `DECODE_FALLIBLE` and `FALLIBLE`, and for a multi-dispatch function the witness names
  an arbitrary representative (`L2Norm` says `f64` for no reason a reader can see). The plan was to
  collapse them into three consts.

  Checking the signatures says no. `arity`, `null_handling` and `is_fallible` on
  `StrictScalarFnVTable` all take *only* the options, with no input dtypes, while `dispatch` needs
  dtypes to choose. So those three answers **must** be dtype-independent, which means they cannot be
  read off whatever element types a batch picks, which is exactly why a separate declaration has to
  exist. The witness is not redundant bookkeeping; it is the only place those facts can live.

  Given that, types beat consts. With types, dense-safety and fallibility are *derived* from the
  element types, so the only available mistake is a witness that disagrees with the dispatch, and that
  is a build error. With three hand-written consts an implementor could state a fact wrongly *and*
  visit consistently with their mistake. Converting would be a notation change that removes a
  derivation, not a fragility fix. Left alone, with the reason now recorded on `ArgsWitness` so the
  next reader does not re-open it.

  What is left of the original complaint is presentational: the arbitrary representative reads oddly.
  A doc line on each multi-dispatch implementor saying why the width shown is arbitrary is the whole
  fix.
- **`InputElement` is an open trait with required consts.** Adding `DECODE_FALLIBLE` broke every
  out-of-crate element (`TensorRow`) until updated. If elements are a real extension point for other
  crates, `DENSE_SAFE` / `DECODE_FALLIBLE` should carry conservative defaults.
- **`DENSE_SAFE`'s doc guidance is subtly wrong for lists.** It says `false` for "any element that
  follows an offset," but a list element *is* dense-safe, because list arrays validate
  `offsets[i] + sizes[i] <= elements.len()` for every row including nulls. Following the doc literally
  would put `list_length` on `Filter` and lose its encoding fast paths.

---

## What the ports bought

**Not line count.** That was the first justification I reached for and it does not hold up: `row/` is
514 code lines and `strict/` is 269, against roughly 470 lines saved across six kernels. Near
break-even. Nor is it bug fixes, since none of the three extracted problems is a live miscompute on
`develop`.

**It is `unsafe`.** Every hand-written kernel in `vortex-tensor` ended the same way:

```rust
// SAFETY: The buffer length equals `len`, which matches the source validity length.
Ok(unsafe { PrimitiveArray::new_unchecked(buffer, validity) }.into_array())
```

A kernel that computes its own values *and* carries its input's validity has to assert that the two
lengths agree, and the only tool for that is `new_unchecked`. The framework never pairs them:
[`OutputElement::build`] returns a non-nullable column, and the strict lifting applies validity
afterwards by masking. The invariant stops being asserted and becomes unrepresentable.

Counting production `unsafe` blocks, test modules excluded:

| function | layer it moved to | `unsafe` on `develop` | `unsafe` now |
| --- | --- | --- | --- |
| `l2_norm` | `RowFn` | 1 | 0 |
| `inner_product` | `RowFn` | 3 | 0 |
| `cosine_similarity` | `RowFn` | 3 | 0 |
| `l2_denorm` | `RowFn` (was `StrictScalarFnVTable`) | 8 | 6 |

**This started as a controlled experiment and the control has since been ported, so read it in two
stages.** For most of this branch's life `l2_denorm` stayed on `StrictScalarFnVTable` and held all 8 of
its blocks while the three functions that moved onto the row layer lost all of theirs. Same crate, same
reviewers, same standards, so the row layer was what removed them rather than the strict lifting or the
port itself. That is the inference the control bought, and it is still the argument.

`l2_denorm` then moved onto the row layer too, via `OutputSink`, and dropped to 6. The two it lost are
exactly the memory-safety ones on its kernel path, which is the pattern the other three showed. Of those
two, one (`FixedSizeListArray::new_unchecked` in the constant-norms path) is attributable to the port and
one (`PrimitiveArray::new_unchecked` in `build_tensor_array`) is an independent cleanup noticed along the
way. Its 6 remaining blocks are a different kind and are not the row layer's business: four call
`L2Denorm::new_array_unchecked`, an `unsafe fn` guarding the *semantic* unit-norm invariant rather than
memory safety, and two are buffer pushes in `normalize_as_l2_denorm`, a helper that is not a scalar
function.

`develop`'s `l2_norm` also hand-rolled a 25-line constant-array fast path that the strict lifting now
does generically for every function, and computed its output nullability by hand.

This is the justification to carry onto a clean branch. It also bounds the claim: a `vortex-tensor`
local helper owning the same invariant would remove the same `unsafe`, so what earns the *generic*
placement in `vortex-array` is that `vortex-geo`'s three predicates and `byte_length` use it too,
over three different element types. Two downstream crates plus core is the second-caller test met, not
anticipated.

### What it costs

Removing that `unsafe` is not free, because `new_unchecked` was buying something: the old kernel paired
its freshly built buffer with the input's validity in one step, so a nullable input cost it nothing
extra. The framework builds a non-nullable column and the lifting applies validity afterwards, which
for `Validity::Array` means materializing a mask and running a separate pass.

That pass is `O(rows)` while the kernel is `O(rows * width)`, so width amortizes it. Measured on
`vortex-tensor/benches/l2_norm.rs`, 16384 rows, `fastest` column:

| width | non-nullable | nullable | cost of the extra pass |
| --- | --- | --- | --- |
| 2 | 68.87 µs | 70.44 µs | +2.3% |
| 32 | 241.4 µs | 243.9 µs | +1.0% |
| 256 | 2.513 ms | 2.529 ms | +0.6% |

So 1 to 2% on nullable input, worst at the narrowest vector anyone would store, and nothing at all on
non-nullable input where no mask is applied. Trading that for eight memory-safety `unsafe` blocks is the
right side of the deal.

These figures are near this machine's noise floor and should be re-confirmed on quieter hardware before
being quoted. The larger measurements in these notes (the 5.7x `like` cache loss, the 8 to 11% `FnMut`
tax, the 2x width-2 per-row cost and its removal, the 2x `l2_denorm` sink win) are well clear of it.

### The like-for-like comparison, and the per-row cost that was hiding in it

The table above compares the framework against itself, so it isolates the masking pass but says nothing
about the rest of the machinery. `PrePortL2Norm` in the same benchmark closes that: a bench-local
`ScalarFnVTable` running the identical arithmetic, indexing the flat slice directly into a `Buffer` and
attaching validity in one step.

This measurement found a real defect in the tensor element, and the diagnosis recorded here first was
wrong in a way worth keeping visible.

**What was measured, and the wrong inference.** `fastest` column, non-nullable, 16384 rows:

| width | framework | pre-port | delta |
| --- | --- | --- | --- |
| 2 | 68.85 µs | 32.85 µs | **2.10x slower** |
| 32 | 266.6 µs | 255.5 µs | +4% |
| 256 | 2.564 ms | 2.512 ms | +2% |

The gap in absolute terms is 36 µs at width 2 and 11 µs at 32, and the conclusion drawn was "a cost that
shrinks as total work grows is a constant being amortized, so the framework carries tens of microseconds
of fixed per-batch setup." That reasoning does not hold. 36 µs over 16384 rows is 2.2 ns/row, which is a
*per-row* cost; it stops showing at width 32 because the kernel there is memory-bound and absorbs extra
CPU work in its stalls. Reading "shrinks with width" as "fixed per batch" skipped dividing by the row
count.

**The actual cause was one per-row accessor, in the tensor element.** `TensorRow::get` called
`FlatElements::row::<T>(i)`, which per row re-derived its typed slice: a ptype comparison against the
stored `PType`, a host-buffer downcast out of the buffer handle, a length division, and then two range
indexings with a bounds check each. All of it loop-invariant except the offset. This is exactly the
hidden-cost-accessor pattern the repository guidelines warn about, and it was written into the element
rather than found in the framework.

The fix types the column at decode time instead of per row. `TensorRow<T>` is already generic over `T`,
so its `Column` can be a `Buffer<T>` plus a stride, and `get` becomes one multiply and one range index
into a typed slice. `FlatElements` keeps its untyped `row` for the callers that read a handful of rows.

**After, same bench, same run:**

| width | framework | pre-port | delta |
| --- | --- | --- | --- |
| 2 | **33.32 µs** | 32.83 µs | **parity, 1.01x** |
| 32 | **227.4 µs** | 258.9 µs | framework **1.14x faster** |
| 256 | **2.422 ms** | 2.522 ms | framework **1.04x faster** |

The pre-port column is stable across both runs (32.85 then 32.83 µs at width 2), which is what makes
this comparison trustworthy; only the framework side moved. `l2_denorm` gained the same way, from
88.0 µs to 48.9 µs at width 2, since it reads its tensor argument through the same element.

Three things follow.

**The row layer was never the cost.** The 2x was one accessor in one element implementation, and the
generic machinery around it (the visitor, the witness, the strict lifting's bookkeeping, `reduce_encoded`'s
probe, the dispatch width match) does not measurably show up at 16384 rows. The planned decomposition
into "strict lifting versus row layer" is moot: neither was it.

**An element is a performance-critical surface, and nothing in the framework says so.** `InputElement::get`
is documented as needing to be `O(1)`, which `FlatElements::row` technically was. `O(1)` is the wrong
contract; the right one is that `get` must not repeat work that is constant across the batch, because it
is the one function called once per row. `decode` exists precisely to hold that work, and the element
vocabulary's whole promise (anyone can add an element in their own crate) means this trap is now
available to every future implementor.

**The framework being generic is what let one fix pay out twice.** `l2_norm`, `inner_product`,
`cosine_similarity` and `l2_denorm` all read tensor rows through this element, so a single change moved
all four. That is the case for the shared layer stated in performance terms rather than in line counts.

### What the harness actually costs, from the optimized IR

The measurements above say the harness is free at 16384 rows. Reading the post-optimization LLVM IR says
*why*, and settles whether more `#[inline]` would buy anything. Emitted with
`cargo rustc --release -p vortex-tensor --lib -- --emit=llvm-ir -Cdebuginfo=0`, reading the `l2_norm` f64
arm.

**The whole stack is already one function.** `execute_row_loop`, `ElementTuple::get` and the row closure
have no `define` of their own anywhere in the module. They survive only as basic-block *labels* carrying
`.exit.i.i.i…` suffixes about sixteen `.i` deep, which is inline-depth notation: the engine's
`ScalarFnVTable::execute`, `execute_dense`, `execute_strict`, `dispatch`, `RowVisitor::visit`,
`execute_row_loop`, `A::get` and the closure are all inlined into a single body. Adding `#[inline]`
anywhere on that path cannot help, because nothing on it is still a call.

**Per batch the harness leaves five calls**, each correctly placed outside the loop: one
`ArgColumn::decode` per argument, one `tensor_element_ptype` for the width match, one `reduce_encoded`,
one `OutputElement::build` after the loop exits, and the output allocation.

**Per row it leaves this, and nothing else:**

```llvm
%row   = phi i64 [ 0, %preheader ], [ %next, %loop_latch ]
%next  = add nuw i64 %row, 1
%start = mul i64 %row, %stride            ; ArgColumn's stride, fused with list_size
%end   = add i64 %start, %list_size
%ovf   = icmp ult i64 %end, %start        ; the two halves of one slice range check
%oob   = icmp ugt i64 %end, %len
br i1 (or %ovf, %oob), label %slice_index_fail, label %body   ; cold side out of line
%rowp  = getelementptr inbounds nuw double, ptr %elements, i64 %start
%endp  = getelementptr inbounds nuw i8, ptr %rowp, i64 %list_size_bytes
...                                        ; element loop, 8x unrolled
%out   = getelementptr inbounds nuw double, ptr %values, i64 %row
store double %result, ptr %out
```

About ten integer ops and one always-taken branch. The element loop underneath is 8x unrolled with a
serial `fadd` chain (LLVM correctly refuses to reassociate the float sum) terminating on `icmp eq ptr`
against `%endp`, which is what a hand-written `iter().map(|x| x * x).sum().sqrt()` compiles to: the
`Elem<'a> = &'a [T]` GAT is fully scalar-replaced, and the slice iterator becomes pointer bumping at
fixed byte offsets.

**The one removable cost is not worth removing.** The surviving per-row branch is the range check on
`&elements.as_slice()[start..start + list_size]`. LLVM cannot hoist it because nothing tells it
`len == rows * list_size`. Eliminating it means `get_unchecked`, and this framework's stated value is
removing `unsafe` from kernels, so buying back a perfectly-predicted branch with an unchecked index is
the wrong direction. It is also already hidden: at width 2 the row's `sqrt` alone has longer latency than
the whole index computation.

LLVM also unswitched the row loop on `list_size == 0` and emitted a zero-width specialization that stores
`0.0` per row. Harmless, and a sign the loop was simple enough to reason about completely.



[`OutputElement::build`]: vortex-array/src/scalar_fn/row/element/mod.rs

Production lines, before and after:

| function | layer | before | after |
| --- | --- | --- | --- |
| `byte_length` | `RowFn` (fixed) | n/a | 23 (impl) |
| `list_length` | `StrictScalarFnVTable` | 189 | 143 |
| `not` | `StrictScalarFnVTable` | 76 (impl) | 53 (impl) |
| `list_sum` | `StrictScalarFnVTable` | 78 (impl) | 56 (impl) |
| `l2_norm` | `RowFn` (width) | 254 | 96 |
| `inner_product` | `RowFn` (width) | 277 | 112 |
| `cosine_similarity` | `RowFn` (width) | 309 | 203 |
| `l2_denorm` | `RowFn` (width, sink) | 731 | 618 |
| geo x 3 | `RowFn` (fixed) | 51 each (impl) | 15 each (impl), plus one shared element |

Nothing outside the functions' own crates changed: the `L2DenormScheme` compressor and every
`ExactScalarFn` matcher are untouched, because the encoding-aware push-downs key off the function
*type* rather than its vtable layer.

The line-count case does not close on its own. The framework is ~1670 production lines (up from ~1510
before the sink, which added `result.rs`, `sink.rs` and a second visit path) and removes ~870 across the
ported functions, so **net this branch adds lines**, amortizing around the fourteenth function against
~20 strict candidates in the tree. To be honest, the case for merging is the marginal
cost of the *next* function (~15 lines, and the invariants above enforced rather than reviewed), plus
the correctness the type-derived properties buy, rather than the diff.

---

## Measurements

`vortex-array/benches/byte_length_element.rs`, element choice for `byte_length`, whole-execution
medians:

| input | `BytesLen` | `Bytes` | |
| --- | --- | --- | --- |
| 64Ki non-inlined rows | **206 µs** | 256 µs | 24% faster |
| 64Ki inlined rows | **207 µs** | 215 µs | 4% faster |

`vortex-array/benches/strict_validity.rs`, how the `Dense` path applies validity, same kernel in both
arms:

| | `lazy` | `eager` | |
| --- | --- | --- | --- |
| 64Ki, one call | **9.0 µs** | 75.3 µs | 8.3x faster |
| 1Mi, one call | 1.357 ms | 1.357 ms | parity |
| 64Ki, chain of 3 | **28.3 µs** | 30.6 µs | 7% faster |

`Validity::and` is already lazy, so the conjunction is never materialized to be applied. Only
`NullHandling::Filter` needs positions, and only it pays for them.

`not`, word-wise kernel against the row loop it would have if it were a `RowFn` (release, identical
outputs asserted):

| len | word-wise `!` | row loop + `bool::build` |
| --- | --- | --- |
| 64Ki | 927 ns | 376 µs (**406x**) |
| 1Mi | 10.3 µs | 5.83 ms (**569x**) |

This is why `not` is a columnar `StrictScalarFnVTable` rather than a row function.

---

## Rejected alternatives

- **A wrapper type instead of a blanket impl** (`Strict<MyFn>`): forces churn at every call site,
  meaning matchers, kernel registrations, and expression constructors. The blanket impl means a port
  edits only the function's own impl block.
- **A `row_family!` macro, a per-crate GAT family, or a framework GAT family**: three encodings of
  "element types as a function of the width," all paying for the same limit (the width bound has to
  appear literally in a GAT), so each width class needed its own trait *and* adapter. The rank-2
  visitor replaces the whole lineage with one non-generic trait method and no generated code.
- **`ElementwiseFn` as a third trait**: subsumed by `RowFn` with a constant dispatch, see above.
- **One `RowFn` with defaulted `dispatch` and `apply`**: converts "define nothing" from a compile
  error into a runtime panic.
- **Renaming `StrictScalarFnVTable` to `TotalFnVTable`**: the trait admits non-total members on
  purpose, so the name would be wrong.
- **An `is_total` method feeding a derived `validity`**: a new concept to compute what a function can
  state directly. Mirroring `validity` with a `None` default makes the unsound answer the one that
  takes work.
- **Macro-generated per-type constructors**: a bespoke API per function, where the general
  `ScalarFnFactoryExt::try_new_array` is what every other scalar function already uses.
- **A separate `FallibleElementwiseFn`**: an associated return type (`ApplyResult`) costs one line per
  function instead of a whole trait and a spent coherence slot.

## Null strategies and the non-strict frontier

The question that opened this chapter: with the strict trait retiring into a private lifting under
`RowFn`, could the row framework also serve non-strict functions, where the kernel sees each input
as an `Option` and owns null semantics itself? The prior expectation was "probably not useful or
performant, but worth establishing why." The answer splits into three verdicts, one per axis, and
the investigation surfaced a fourth result nobody asked for that is worth more than the question.

Method: a survey of every non-strict `ScalarFnVTable` impl in the workspace plus every consumer of
`is_strict` and `validity()`, and a working prototype (worktree branch `proto/null-strategies`,
2,034-line diff, not for merging) that implemented both a branch-and-skip execution strategy and a
`Nullable<E>` input element, benchmarked on 65,536-row batches at null densities from 0% to 90%.
All 435 vortex-array scalar_fn tests and 223 vortex-geo tests pass with the prototype strategy both
off and on, including new hostile tests (out-of-bounds views and poison divisors behind null rows)
proving the kernel never runs behind a null.

### Verdict 1: null-visible inputs have no customer, and now we know the price

The survey found 15 non-strict functions. Thirteen are cheap columnar mask algebra or pure
structure. The canonical case is Kleene `AND`: a fused kernel computing values and validity
together at roughly six bitwise ops per 64 rows, with validity `(lv & rv) | (lv & !l) | (rv & !r)`.
The prototype measured a row-function Kleene `AND` over `(Nullable<bool>, Nullable<bool>)` against
it: **250x to 1,030x slower** depending on density. That is the honest price of spelling bitwise
logic one row at a time, and no framework design recovers it.

The remaining two, `RowEncode` and `RowSize` in vortex-row, are the only genuinely expensive
null-visible per-row kernels in the tree, and they are excluded by something the Option tier does
not touch: they are variadic over heterogeneous column types with a shared per-row write cursor,
which the fixed-arity tuple witness cannot express. Null-visible inputs alone unlock nothing.

Four functions (Kleene `AND`/`OR`, `zip`, `case_when`, `list_contains`) have **value-dependent
output validity**: `false AND null` is a *valid* `false`. For these no validity expression over
child validities exists even in principle, so the lifting's derivations (validity expression, mask
motion, dictionary push-down eligibility) are unavailable by definition rather than by
implementation gap. Any future Option-input tier must let the kernel author value and validity
together, which is to say it must be a different trait, not a mode of this one.

What `is_strict = false` forfeits is exactly enumerable: the dictionary values push-down
(`arrays/dict/compute/rules.rs`), the dict-layout below-decode push-down
(`vortex-layout/src/layouts/dict/reader.rs`), and, when `validity()` is also `None`, lazy validity
on an unexecuted `ScalarFnArray` degrades to executing the kernel to read its nulls. Nothing in
vortex-scan, vortex-file, or the engine integrations consumes strictness.

Mechanically, `Nullable<E>` works exactly as sketched: `Elem<'a> = Option<E::Elem<'a>>`, decode
materializes the validity mask once, `get(i)` consults it, `DENSE_SAFE = true` by construction.
Niche packing is free for every by-reference element (`Option<&[u8]>`, `Option<&str>`,
`Option<&[T]>`, `Option<&Geometry>`, `Option<bool>` all compile-time asserted same-size) and
doubles every by-value primitive, which are precisely the elements that were already dense-safe
and never needed a strategy. The prototype's geo `contains` over `(Nullable<GeometryRow>, const)`
tracked branch-and-skip within 2-8%, so the shape is viable for a kernel that wants null
visibility for semantic reasons. Nothing in the tree does. **Do not build it; keep the survey's
constraint list for whenever a real variadic or null-visible demand shows up.**

### Verdict 2: Option outputs inside the strict tier are the real demand

Strictness is a subset bound, `valid(out) ⊆ valid(in)`, so a kernel that turns a valid row into a
null is still strict, and the strict lifting already keeps kernel-produced nulls, unioned with the
lifted ones. What excludes such functions from `RowFn` today is only the all-valid-output rule on
`OutputElement`. Two in-tree functions are shaped exactly like this: `list_sum` (a valid empty
list sums to null; the module doc names it as the canonical exclusion) and `variant_get`
(expensive per-row path traversal where a missing path yields null). The extension is small and
local: an `Option<T>` output form whose element dtype is nullable and whose build sets validity,
`RetWitness` gaining a nullability bit alongside `FALLIBLE`, and the derived `validity()` moving
from `union_child_validities` to `None` for such functions, which costs them lazy validity but is
already the status quo for both named candidates. `is_strict` stays `true`. **This is the piece
worth building.**

### Verdict 3: branch-and-skip, the result nobody asked for

Today the derived null handling is binary: `Dense` (run over garbage, mask after) when every
element is dense-safe and the kernel infallible, else `Filter` (filter every input to the
conjoined-valid rows, run, scatter back). The prototype added the missing third strategy:
materialize the conjoined mask once, run over the *unfiltered* inputs visiting only set rows
word-at-a-time (`BitBuffer::for_each_set_index`), pre-fill the output with garbage, mask exactly
as Dense does. Fallible kernels stay sound because apply never runs behind a null.

Measured against Filter at 65,536 rows (divan fastest, two runs):

| workload | 1% nulls | 10% | 25% | 50% | 90% |
| --- | --- | --- | --- | --- | --- |
| `byte_length` at `Bytes` (cheap kernel) | branch 1.8x | 2.6x | 3.8x | 4.7x | **5.9x** |
| geo `contains`, one nullable operand | branch 1.07x | 1.11x | 1.18x | 1.11x | filter 1.38x |
| geo `contains`, two nullable operands | branch 1.06x | even | filter 1.2x | filter 1.9x | filter 11.3x |

For the cheap kernel Filter never wins: at even 1% nulls, filtering the input plus scattering the
output costs more than the entire branch-side loop. For the expensive kernel the governing
quantity is the **surviving-row fraction**: branch pays O(n) decode regardless, Filter pays
O(survivors) decode plus filter and scatter. Geo's ablation makes the mechanism explicit: filter
plus scatter are under 4% of `contains`' total, so Filter's entire advantage at sparse validity is
the shrunken arrow-export-and-parse, while for `byte_length` those same two steps are 20-40% of
Filter's total and pure waste. Crossover lands near 50-75% surviving rows for one nullable operand
and lower with two (the conjoined fraction shrinks quadratically).

The strategy is invisible to function authors: it slots under the existing derived null handling,
selectable per batch from `Mask::true_count`, with Filter kept for the sparse tail. **This is now
implemented on this branch** (see "Adaptive null strategy, as shipped" below); the rest of this
section records the prototype evidence that justified it. The prototype
also validated the two supporting pieces: a null-tolerant `decode_branch` on `InputElement`
(defaulting to plain decode, correct for bulk canonicalizations) and `OutputElement::garbage()`
for pre-fill. Production caveats recorded in the prototype report: `reduce_encoded` is not
consulted on the branch path, sinks fall back to Filter, the toggle must become per-execution and
cost-based, and geo's null-tolerant decode covered Point and Polygon only, still paying a
full-length arrow export that a run-slicing decode would shrink. The prototype's conclusion, since borne out: `Bytes`-element functions were paying the
Filter tax on every nullable batch, and most of it is recoverable.

### Adjacent findings, recorded so they are not relearned

- `Between::validity` declares the strict three-way conjunction while its fallback execute path
  joins two comparisons with Kleene `AND`; with per-row nullable bounds the lazy validity and the
  executed result disagree (a valid `false` reported as null). Pre-existing on develop,
  independent of this work, slated-for-removal expression; deserves an issue.
- `not` is already at the optimum reachable through the current ownership model: `to_bit_buffer()`
  is a handle clone, the source array keeps the buffer shared, so in-place negation (a real 19% on
  uniquely owned buffers) is unreachable without redesigning `ExecutionArgs` ownership. Encoded
  NOT flows through `NotReduce` (Constant, Sparse) and generic per-encoding push-down (Dictionary,
  RunEnd) at 13-24x below canonical cost; `NotKernel` has no implementations and looks like dead
  code. The three columnar ports of the retired strict trait revert entirely.
- The strict lifting's small-batch overhead is generic prelude bookkeeping (collect inputs,
  compute the declared dtype, conjoin validity), not any single avoidable allocation; ablations
  including SmallVec found nothing independently beneficial, and the earlier -10%-at-100-rows
  reading did not reproduce uniformly. The row layer can eventually monomorphize the prelude over
  its compile-time arity (`[ArrayRef; N]` via the tuple witness), which is the only structural
  answer if small batches ever matter.

## Adaptive null strategy, as shipped

Branch-and-skip is implemented as a third null strategy, chosen per batch by the lifting. Nothing
about a function's definition changes: the row layer already derived `Dense` or `Filter` from the
element types, and `Filter` now names a *contract* (the kernel never sees a row null in any input)
rather than a mechanism. Two mechanisms satisfy that contract, and the lifting picks between them
where the conjoined mask is materialized.

The selection rule needs one fact the framework cannot infer, so elements state it:
`InputElement::DECODE_SHRINKS_WHEN_FILTERED`, defaulted `false`, is `true` for an element whose
decode parses every row (geometry from coordinate storage) and `false` for a bulk canonicalization
(bytes, bools, primitives). Getting it wrong is a performance bug, never a correctness bug.
`ElementTuple` ORs it across arguments, the witness check pins it like dense-safety and
fallibility, and the rule is:

```text
branch-and-skip, UNLESS some argument's decode shrinks when filtered
                 AND fewer than BRANCH_MIN_SURVIVING_FRACTION (0.75) of rows survive
```

Two supporting hooks: `InputElement::decode_null_tolerant` (defaults to the ordinary decode, sound
because the branch loop never resolves an unset row, so hostile bytes behind a null are never
touched) and `OutputElement::placeholder` (the pre-fill written behind nulls, masked before anyone
observes it). Geo overrides the decode for Point and Polygon; other geometry types report
unsupported and the selection falls back to Filter, which is tested rather than asserted in a
comment. Sinks stay on Dense/Filter, documented at the visitor. `reduce_encoded` runs on the
branch path over the *original* encodings, which is strictly better for encoding fast paths than
Filter's canonical copies, and its contract doc now states the row count differs per strategy.

The original forced-filter, forced-branch and auto measurements used 65,536 rows on a shared 4-vCPU
VM:

| workload | 1% | 5% | 10% | 25% | 50% | 90% |
| --- | --- | --- | --- | --- | --- | --- |
| `byte_length` at `Bytes`, auto over filter | 5.0x | 5.3x | 5.8x | 4.0x | 4.5x | 6.3x |
| geo `contains` x const, auto picks | branch | branch | branch | branch | filter | filter |
| geo `contains` x column, auto picks | branch | branch | branch | filter | filter | filter |

Those historical rows justified shipping branch-and-skip, but they no longer calibrate the global
threshold. The controlled x86 AVX-512 rerun used a Ryzen 9 7950X pinned to CPU 4, TSC timing, a
performance governor, 60 samples for 2-4 seconds per arm, and two runs. Its representative medians
were:

| workload | auto | branch | filter | verdict |
| --- | ---: | ---: | ---: | --- |
| one nullable, 50% nulls | 5.999-6.050 ms | 5.560-5.642 ms | 6.026-6.049 ms | auto filters, branch is 6-8% lower latency |
| two nullable, 10% nulls | 10.40-10.48 ms | 10.49-10.60 ms | 10.20-10.34 ms | auto branches, filter is 2.5-2.8% lower latency |
| two nullable, 25% nulls | 7.502-7.678 ms | 9.156-9.285 ms | 7.588-7.749 ms | auto correctly filters; filter is 1.21-1.22x faster than branch |
| two nullable, 90% nulls | 277.1-277.4 us | 3.232-3.253 ms | 277.7-278.5 us | auto matches filter; filter is about 11.6x faster than branch |

The two misses point in opposite directions. A 50% surviving one-element decode still favors
branch, while an approximately 81% surviving two-element decode already favors filter. A single
threshold against the conjoined survivor fraction therefore cannot represent both decode cost and
arity. Replace it with per-element/arity inputs or a small estimated-cost comparison when this work
moves onto production branches. Batch size remains an unmeasured input to that model.

Verified independently of the implementing agent: 3,441 tests pass across vortex-array and
vortex-geo (17 new: hostile out-of-bounds views behind nulls, a fallible kernel with poison
divisors behind nulls in one and both operands, conjoined-mask honoring, constant operands, real
errors still propagating, geo filter-versus-branch agreement, the unsupported-geometry fallback,
and six selection-rule cases), vortex-tensor's 164 pass unchanged, clippy `--all-targets
--all-features` is silent on both crates, and fmt and whitespace are clean.

Open items, none blocking: the branch fallback probes `reduce_encoded` twice when the dispatch
turns out unsupported (cheap encoding check, no in-tree function affected since every
`reduce_encoded` implementor is a dense-path tensor function); geo's null-tolerant decode still
arrow-exports the full column, and slicing runs of valid rows would blunt Filter's sparse-validity
advantage enough to retire the threshold for geo; the fallible branch loop pays one `is_none`
check per set row after the first error because `for_each_set_index` cannot early-return.

## The strict trait, deleted

`StrictScalarFnVTable` is gone. Not made private: deleted, with its lifting kept as private
machinery under `vortex-array/src/scalar_fn/row/lift.rs`. The chain is now `RowFn` ->
`ScalarFnVTable`, one blanket impl, no intermediate trait.

Three things converged on that. First, reverting the columnar ports left the trait with exactly one
implementor, the blanket impl over `RowFn`, and a trait with one impl is indirection rather than
abstraction. Second, the mirroring tax existed only because that blanket impl occupied the
`ScalarFnVTable` slot: `reduce` and `validity` were forwarded so a strict function could override
them despite being unable to implement `ScalarFnVTable` itself. `RowFn` keeps `validity`, because
all-valid outputs make it the child conjunction, and the `reduce` mirror went with the trait since
no adopter ever used it. Third, the naming objection a local review raised was real and is now
moot: `is_strict` names the semantic property `valid(f(x)) ⊆ valid(x)` that pushdown consumes,
while the trait demanded the *operational* property that a kernel may run over the garbage behind a
null row or over a filtered copy. Those are independent, `Bytes` being strict and not dense-safe,
so the trait was named for the wrong one of the two.

What replaced each member: `execute_strict` and `execute_strict_branch` are the two closures
`Batch::execute` takes, `decode_shrinks_when_filtered` is a `Batch` field read off
`ElementTuple::DECODE_SHRINKS_WHEN_FILTERED`, `return_element_dtype` is what a visit returns before
`ScalarFnVTable::return_dtype` widens it, `null_handling` is `row_null_handling` over the witnesses,
and options serde is `RowFn::Options: PersistableOptions` delegated from the blanket impl.
`Batch` carries one batch's facts (id, arguments, collected inputs, conjoined validity, declared
return dtype, null handling, and the decode-shrinks flag) and takes the kernel as closures rather
than through a trait, which is the point: there is no second implementor to name.

The one behaviour deliberately dropped is the runtime rejection of `Dense` paired with a fallible
kernel. `row_null_handling` derives the pairing from the same witnesses `is_fallible` reads, so the
combination cannot be constructed, and the requirement now lives in `NullHandling::Dense`'s doc
pointing at the derivation. Four tests went with the trait: three described a strict kernel that
returns nulls of its own (`list_sum`'s shape), which no `RowFn` can be until the `Option` output
form of open item 3 exists, and one pinned the `reduce` mirror.

`PersistableOptions` survives with `EmptyOptions` as its only implementor, since every row function
in tree uses it. That is a bound on `RowFn::Options` rather than a speculative trait, and the
reverted `list_sum` port is what removed its second implementor.

If a non-row columnar kernel ever wants the lifting, extract the trait then, named for the lifting
contract rather than for strictness, with that kernel as its first user.

## Sink-only execution, the final prototype

The last executor revision collapses every row function onto one primitive:

```rust
visitor.visit_prepared_into::<Args, Sink, _, _>(
    |constant_args| prepare(constant_args),
    |state, args, output| write_one_row(state, args, output),
)
```

The ordinary case uses unit preparation and `ElementSink<T>`. A tensor uses `TensorSink<T>` so the
input dtype can determine the runtime row width. A future string transform can own one batch-wide
builder. These are not different executor modes, so the API no longer gives them different visit
methods.

### Why the return witness disappeared

A returning row closure needed a return witness before dispatch so `return_dtype` and fallibility
could be derived without knowing which dtype arm dispatch would select. Once every closure writes
through a sink, the sink already answers the output question:

- `sink_dtype(args)` supplies the non-nullable element or runtime-shaped dtype.
- `with_capacity` allocates once for the batch.
- `rows` borrows the loop-local storage once.
- `row_count_matches` proves the output bound once.
- `row` hands one slot into the closure.
- `finish` builds the column and interprets any deferred error.

`RowFn::ArgsWitness` remains load-bearing because arity and input decode properties are needed
before dispatch. `RowFn::FALLIBLE` remains because `ScalarFnVTable::is_fallible` is queried without
input dtypes. There is no analogous need for a return witness.

The closure stays `Fn`, not `FnMut`. An earlier sink design captured `&mut Sink` in the closure and
measured 8 to 11% slower because the mutable capture blocked loop vectorization. The executor now
owns the sink, borrows its rows once, and passes a row slot as an ordinary argument.

### Errors without a per-row result branch

`SinkResult` has three implementations:

- `()` for an infallible write.
- `VortexResult<()>` for an error that must exit immediately.
- `DeferredError` for a row that can write a legal provisional value and report failure after the
  loop.

Checked integer addition is the motivating deferred case. Its sink writes the wrapping sum, each
row returns a word whose sign bit means overflow, and the executor OR-reduces those words. `finish`
returns the overflow error only when the final word has its sign bit set. No `Result` discriminant
or conditional error branch is required per row.

Nullable dense execution needs one extra rule. Garbage behind a null may overflow even when every
valid row succeeds. When dense execution finishes with a deferred error, the lifting materializes
the conjoined validity and retries only valid rows. A successful retry proves the first error came
only from discarded rows; a second deferred error is real. This preserves strict null propagation
without giving up the dense vector loop on the common path.

This is deliberately narrow. Parsing, allocation, and any computation that cannot produce a legal
provisional row still returns `VortexResult<()>` and receives valid-row-only execution.

### Skipped rows are a sink property

`OutputSink::SUPPORTS_SKIPPED_ROWS` replaces the earlier blanket statement that sinks cannot use
branch-and-skip. `ElementSink<T>` pre-fills `OutputElement::placeholder` and supports skipped rows.
A custom sink may do the same, or decline and let the lifting filter and scatter. The semantic
contract remains that skipped values are legal but arbitrary and are masked before the result
escapes.

### Final executor measurements and IR

The authoritative `row_fn_executor` run used 65,536 `i64` rows, 100 samples, a one-second minimum
per arm, TSC timing, CPU 4, and a performance governor on the Ryzen 9 7950X. Each cell is the range
across two runs as fastest / median:

| workload | specialized | sink-only `RowFn` | specialized / `RowFn` |
| --- | ---: | ---: | ---: |
| checked add, two columns | 131.5-132.3 / 132.3-133.3 us | 128.4-129.6 / 129.5-130.9 us | 1.021-1.024x / 1.018-1.022x |
| checked add, column and constant | 16.90-16.93 / 17.10-17.21 us | 13.82-13.85 / 14.04 us | 1.222x / 1.218-1.226x |
| checked add, nullable columns | 133.8-134.8 / 136.1 us | 128.4-128.7 / 130.6-131.8 us | 1.042-1.047x / 1.033-1.042x |

The native release IR has `<8 x i64>` vector error-word accumulators and
`llvm.vector.reduce.or.v8i64`. The two-column assembly is four-way unrolled over AVX-512 `zmm`
registers, producing 32 `i64` rows per iteration with four `vpaddq` instructions. Overflow bits
accumulate through vector xor/ternary-OR operations and reduce after the loop; there is no per-row
result discriminant or error branch. The specialized arm remains benchmark-local, and no production
deferred-error user exists yet.

Other final diagnostic medians:

- `strict_validity` lazy versus eager stayed within 2% across 65,536 and 1,048,576 rows, including
  a chain of three calls.
- `byte_length_element` found `BytesLen` 1.410-1.411x faster by median than resolving a byte slice
  for long strings and 1.097x for short/inlined strings at 65,536 rows. This justifies the element
  choice but is not a production benchmark.
- `null_strategy_bytes` auto matched branch-and-skip; at 90% nulls it took 24.95 us against
  175.4 us for filter-and-scatter.
- Geo auto broadly tracks branch at dense validity and filter at sparse validity, but the controlled
  x86 run found the two threshold misses recorded above. The full forced-strategy matrix remains an
  implementation diagnostic, not permanent CodSpeed coverage.
- Distinct per-row LIKE patterns took 126.4 us against 26.87 us for a repeated pattern, 4.7x
  slower. That is the measured reason LIKE remains a stateful columnar implementation.

### Durable benchmark boundary

Draft PR [#9136](https://github.com/vortex-data/vortex/pull/9136) now owns the stable production
benchmark names. At `bf814bbe02cb` it covers public-path byte length; signed and unsigned add,
including constant and nullable inputs; repeated and distinct LIKE patterns; tensor functions and
the `Normalized` encoding; and geo contains, intersects, and distance with constant and nullable
shapes. It also reduces the expensive overlapping-contains simulation to 1,024 rows and uses
vendored `mimalloc` in allocating binaries.

Do not merge the research harnesses above into that permanent suite. They compare internal
strategies or frozen controls that do not exist on develop. Land #9136 first, then use its identical
benchmark names to gate each production implementation PR through CodSpeed's compiled amd64/AVX2
simulation. Keep local Divan for real wall-clock diagnosis and generated IR for explaining a
regression.

### Final API consequence

Issue 9129's current sketch is obsolete: it still has `RetWitness`, `visit`, `visit_prepared`, and
`visit_into`. Issue 9130 still says sink-backed execution cannot branch-and-skip. Update both before
using their checklists to cut the implementation stack. The prototype to carry forward is:

```text
RowFn
  -> dispatches Args + OutputSink through visit_prepared_into
  -> private Batch lifting chooses dense, branch-and-skip, or filter-and-scatter
  -> ElementSink covers ordinary output
  -> custom sinks cover runtime shape and deferred errors
  -> ScalarFnVTable blanket impl exposes the function
```

Nullable outputs remain separate. A sink can build values plus validity, but doing so invalidates
the unconditional `validity() = union_child_validities` derivation. That semantic change should
land with its first strict non-total user, not inside the initial sink executor.

---

## Final API simplification review

This section supersedes every earlier API sketch in this document. In particular, do not carry
forward `ArgsWitness`, `RetWitness`, `PersistableOptions`, public `NullHandling`,
`DECODE_SHRINKS_WHEN_FILTERED`, or `TensorSink`.

The review started from two constraints. The public API should expose only decisions a function
author can meaningfully make, and the executor should not trust facts fabricated by downstream
implementations. Applying both constraints removed more framework surface without preventing a
function from defining domain-specific rows.

### The final extension boundary

The framework is selectively sealed:

- `RowFn` remains open. It names the function, options, argument names, fallibility, persistence,
  and dtype-based dispatch.
- `InputElement` remains open. This is how a crate adds a new decoder for a geometry, tensor view,
  byte view, or another domain scalar.
- `OutputElement` remains open for ordinary one-value-per-row outputs.
- `OutputSink` remains open for output representations that need their own builder or row state.
- `RowVisitor`, `ElementTuple`, and `SinkResult` are sealed because their implementations assert
  executor facts used by the blanket vtable.

Sealing `ElementTuple` does not seal decoding. The framework supplies tuple recursion for arities 0
through 12, and a function places any open `InputElement` implementation inside those tuples.
Sealing `SinkResult` likewise does not seal output representation. A custom `OutputSink` selects one
of the supplied result behaviors.

This keeps the author vocabulary extensible while avoiding public implementations that can lie
about arity, dense safety, result fallibility, deferred errors, or skipped-row support.

### Dispatch contains its own evidence

`RowFn` no longer has argument or return witnesses. `ARG_NAMES.len()` is the exact arity. The types
selected by `dispatch` carry the remaining evidence:

```text
(InputElement, ...) + OutputSink + SinkResult
  -> arity and decode properties
  -> output representation and dtype
  -> row fallibility and deferred-error word
```

The visitor asserts at compile time that the dispatched tuple arity matches `ARG_NAMES`, a
fallible decoder or result implies `RowFn::FALLIBLE`, and deferred evidence is accepted by the
selected sink. These are implications rather than equalities. A function may conservatively
declare `FALLIBLE = true` while selecting an infallible arm for some dtypes.

This is enough for planning because dispatch is pure in `(options, args)`. It is also simpler than
duplicating the same tuple in a witness and every dispatch arm, then proving that the declarations
agree.

### Persistence follows the function ID

`Options: PersistableOptions` assigned one wire contract to a Rust type. That was the wrong owner.
Two functions may reuse an options type while choosing different encodings or serializability, and
an unregistered function should not invent persistence merely because its options type supports it.

The final `RowFn` therefore owns `serialize` and `deserialize` hooks. Serialization defaults to
`Ok(None)`, and deserialization defaults to an error. Registered tensor and geo functions preserve
their explicit existing formats. The unregistered `NumericBinary` needs no otherwise-unused
serialization implementation for `NumericOperator`.

### One custom sink is enough

`OutputSink` already permits arbitrary internal state. A function that needs two builders defines
one sink with two fields rather than asking the executor to understand pairs of sinks. The same
rule applies to other composite or runtime-shaped results: express the shape inside one sink and
add framework abstraction only after two real users expose shared mechanics.

The public `TensorSink` had no user after `l2_denorm` became the `Normalized` encoding. `l2_norm`,
inner product, and cosine similarity all return scalar rows through `ElementSink`. Removing
`TensorSink` avoids stabilizing roughly 90 lines of runtime-shaped row behavior without preventing
a future tensor-valued function from defining a private sink.

`ElementSink<T>` also no longer needs an `ElementRow` wrapper. Its row is `&mut T`, and a closure
writes with `*output = value`. The sink still pre-fills legal placeholders so branch-and-skip may
leave masked rows untouched.

### Per-argument filtered-decode cost

The aggregate `DECODE_SHRINKS_WHEN_FILTERED` flag was measurably lossy. OR-ing the flag made one
expensive decode indistinguishable from two, even though the x86 data selected opposite mechanisms:

- one nullable geometry argument at 50% nulls favored branch-and-skip; and
- two independently nullable geometry arguments at 10% nulls, about 81% surviving rows, favored
  filter-and-scatter.

`InputElement::FILTERED_DECODE_COST` now defaults to zero, and each tuple adds the costs of all its
arguments. The batch selector uses the following coarse policy:

- cost 0 always branches;
- cost 1 branches at 50% or more survivors; and
- cost 2 or greater branches at 85% or more survivors.

The exact values come from the measured cases rather than a general cost model. There is not yet
enough evidence to distinguish two costly arguments from three, or to make the crossover depend on
batch size. Keep the value additive so a later selector can use that information without another
author-facing API change.

The old public `NullHandling` enum is gone. The executor privately derives `Dense`,
`DenseWithRetry`, or `ValidOnly { filtered_decode_cost }`. Authors declare local safety and cost on
their input/result types, not a global mechanism. `NullStrategy` survives only in the test harness
to force branch-and-skip or filter-and-scatter.

### Deferred errors stay in a loop-local word

The numeric migration confirmed two constraints on deferred error evidence:

- the accumulated word must be no wider than the element type; and
- the accumulator must live in the generated loop, not behind a mutable sink reference.

The sealed `SinkResult` implementations for `bool`, `u8`, `u16`, `u32`, and `u64` preserve both.
Checked multiplication can report discarded high bits directly, LLVM can accumulate those words in
vectors, and `finish` turns the final evidence into the function error. `VortexResult<()>` remains
the separate early-exit form for a row that cannot write a legal provisional value.

### Code generation after the simplification

The final cleanup at `4becc863ae` was compared with parent `53c51d803c` using rustc 1.91.0 and LLVM
21.1.2. Both revisions were cross-compiled with:

```bash
cargo rustc -p vortex-array --bench row_fn_executor --profile bench \
    --target x86_64-apple-darwin -- \
    --emit=llvm-ir -C codegen-units=1 -C target-cpu=x86-64-v3
```

The optimized executor monomorphs were normalized to remove revision-specific symbol names and
metadata. Their vector/reduction block hashes matched exactly for wrapping add through
`ElementSink`, checked add with deferred evidence, and wrapping add through the custom `I64Sink`.

The two wrapping paths retain 256-bit `<4 x i64>` loads, adds, and stores across six vector loop
bodies covering constant and varying inputs. Checked add retains `<4 x i64>` arithmetic, derives
overflow with vector xor/and/compare operations, ORs `<4 x i1>` evidence in the vector loop, and
reduces after the loop. The vector bodies have no calls or panic references. Scalar tails are
present in both revisions.

The production tensor benchmark IR was checked separately for `l2_norm`, inner product, and cosine
similarity. After normalizing SSA and metadata, arithmetic sequences and instruction counts matched
between revisions for both `f32` and `f64`. Their ordered floating-point reductions remain
eightfold scalar-unrolled in both revisions. They were not vectorized before the cleanup, so the
API change did not cause that property.

Native Apple M4 Max `row_fn_executor` timings used 65,536 rows, two alternating revisions, 100
samples, and a 0.5-second minimum per arm. RowFn median deltas ranged from 1.11% faster to 0.94%
slower. Fastest deltas stayed within approximately 0.17%, while specialized controls had median
drift as high as 3.7%. That is no measurable native regression.

This evidence is deliberately bounded. Cross-target optimized IR shows that the API cleanup did
not change the x86_64-v3 hot loops. It cannot establish the runtime effect of the new null selector
on an x86 branch predictor. Re-run the measured null shapes on x86 before changing or declaring the
50% and 85% thresholds settled.

### Required x86 rerun

The next session will run on an x86 machine. It must rerun the production comparison before this
performance record is considered complete. The #9136 benchmark baseline is now on `develop` at
`9a482c0230`, including the public binary, tensor, and geo benchmark binaries used by this work.
Fetch the latest `origin/develop`, record both exact revisions, and compare the branch against
`develop` with the same benchmark names.

Run `binary_ops` and `like` from `vortex-array`. Run `l2_norm`, `inner_product`,
`cosine_similarity`, and `normalized` from `vortex-tensor`. Run `binary_predicates`, `distance`,
`envelope`, and `predicate_bbox` from `vortex-geo`. Use at least two alternating runs per revision.
If the host permits it, pin one core. Report both fastest and median values with the CPU, timer, and
governor configuration.

The stable production binaries are the cross-revision gate because they now exist on `develop`.
The branch-only `vortex-geo` `null_strategies` benchmark remains the forced-policy diagnostic. Run
it on the same x86 host to verify both measured selector decisions: one costly decode at 50%
survivors must select the faster mechanism, and two costly decodes at about 81% survivors must do
the same. Inspect optimized LLVM IR again for any stable regression before changing the API or the
selector.

### Final verification state

The final API state recorded 67 focused RowFn tests, 179 tensor tests, and 230 geo tests. Nightly
formatting passed. Full workspace clippy passed with
`PYO3_NO_PYTHON=1 PYO3_BUILD_EXTENSION_MODULE=1`, required because the host `/usr/bin/python3` is
3.9 while the workspace targets the Python 3.11 stable ABI.

Issues #9128, #9129, and #9130 were updated to this API. The durable public-path benchmark baseline
from #9136 is now in the repository. Earlier statements in this document that those issues or the
baseline still need updating are historical only.
