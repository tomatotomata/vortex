// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks [`normalize`], the norm split `NormalizedScheme` runs before it cascades.
//!
//! Null handling is the axis under measurement. `normalize` has to move the input's nulls onto the
//! array and zero both children at those positions, and there are two ways to reach the norms
//! child: materialize the validity into a `Mask` and branch on it per row while building a second
//! buffer, or hand the whole thing to the `fill_null` kernel. The non-nullable case is the one that
//! decides it, since that is most columns and `fill_null` degrades to a cast there while the mask
//! has to be built regardless.
//!
//! The two dimensions bracket where that cost lands. At `8` the per-row work is visible; at `768`,
//! a typical text-embedding width, the per-element division dominates and per-row differences
//! disappear into it. Row counts are chosen per dimension to keep every case under the 1 ms
//! per-iteration budget in `docs/developer-guide/benchmarking.md`.

#![allow(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
use vortex_tensor::encodings::normalized::normalize;
use vortex_tensor::vector::Vector;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    vortex_tensor::initialize(&session);
    session
});

/// `(dimension, rows, one-in-N rows null)`. A null stride of `0` leaves the column non-nullable.
const CASES: &[(u32, usize, usize)] = &[
    (8, 4096, 0),
    (8, 4096, 2),
    (8, 4096, 16),
    (768, 512, 0),
    (768, 512, 2),
    (768, 512, 16),
];

/// Builds a `Vector` column whose coordinates vary per row, so no row is a zero vector and every
/// row takes the dividing branch.
fn vector_column(dim: u32, rows: usize) -> ArrayRef {
    let elements: Buffer<f64> = (0..rows)
        .flat_map(|row| (0..dim).map(move |i| (row + 1) as f64 * (i + 1) as f64))
        .collect();
    let storage = FixedSizeListArray::new(elements.into_array(), dim, Validity::NonNullable, rows)
        .into_array();

    Vector::try_new_vector_array(storage).unwrap()
}

/// Masks every `null_every`-th row, or leaves the column non-nullable when `null_every` is 0.
fn input(dim: u32, rows: usize, null_every: usize) -> ArrayRef {
    let column = vector_column(dim, rows);
    if null_every == 0 {
        return column;
    }

    let validity = Validity::from_iter((0..rows).map(|i| !i.is_multiple_of(null_every)));

    MaskedArray::try_new(column, validity).unwrap().into_array()
}

#[divan::bench(args = CASES)]
fn normalize_column(bencher: Bencher, case: (u32, usize, usize)) {
    let (dim, rows, null_every) = case;
    let input = input(dim, rows, null_every);

    bencher
        .with_inputs(|| (input.clone(), SESSION.create_execution_ctx()))
        .bench_values(|(input, mut ctx)| normalize(input, &mut ctx).unwrap());
}
