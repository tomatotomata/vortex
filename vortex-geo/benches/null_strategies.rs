// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Null-strategy comparison for the geo `contains` kernel, whose per-row geometry decode is what
//! the selection threshold exists for.
//!
//! Arms: `filter` and `branch` force one strategy through the test-harness seam
//! ([`execute_row_fn_with_strategy`]); `auto` executes the full pipeline and lets the per-batch
//! selection choose, which should track the faster forced arm on both sides of the crossover
//! (branch at dense validity, filter at sparse).
//!
//! Workloads: a column of small polygons CONTAINS a constant point, and polygon column CONTAINS
//! point column with independent nulls on both, each at null densities 0/1/5/10/25/50/90 percent
//! over 65536 rows, nulls placed by a seeded splitmix hash.
//!
//! Run with `cargo bench -p vortex-geo --bench null_strategies`.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::ItemsCount;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::MaskedArray;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::NullStrategy;
use vortex_array::scalar_fn::execute_row_fn_with_strategy;
use vortex_array::validity::Validity;
use vortex_geo::scalar_fn::contains::GeoContains;
use vortex_geo::test_harness::geo_session;
use vortex_geo::test_harness::point_column;
use vortex_geo::test_harness::polygon_column;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(geo_session);

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

const ROWS: usize = 65536;

/// Null densities in percent.
const DENSITIES: &[usize] = &[0, 1, 5, 10, 25, 50, 90];

/// Deterministic pseudo-random value in `[0, 1)` (same generator as `binary_predicates`).
fn unit(i: usize) -> f64 {
    ((i.wrapping_mul(2654435761) >> 8) % 10_000) as f64 / 10_000.0
}

/// splitmix64, for seeded random null placement.
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// A small square (side 2) centered at `(cx, cy)`.
fn square(cx: f64, cy: f64) -> Vec<Vec<(f64, f64)>> {
    vec![vec![
        (cx - 1.0, cy - 1.0),
        (cx + 1.0, cy - 1.0),
        (cx + 1.0, cy + 1.0),
        (cx - 1.0, cy + 1.0),
        (cx - 1.0, cy - 1.0),
    ]]
}

/// [`ROWS`] small squares spread over roughly `[-150, 150)^2`; a handful contain the origin, and
/// each row's verdict is a direct point-in-polygon test.
fn squares() -> ArrayRef {
    let rows = (0..ROWS)
        .map(|i| square(300.0 * unit(i) - 150.0, 300.0 * unit(i + 1) - 150.0))
        .collect();
    polygon_column(rows).unwrap()
}

/// [`ROWS`] points over the same region.
fn points() -> ArrayRef {
    let xs = (0..ROWS).map(|i| 300.0 * unit(i + 7) - 150.0).collect();
    let ys = (0..ROWS).map(|i| 300.0 * unit(i + 8) - 150.0).collect();
    point_column(xs, ys).unwrap()
}

/// The constant point operand, at the origin so some squares contain it.
fn constant_point(ctx: &mut ExecutionCtx) -> ArrayRef {
    let scalar = point_column(vec![0.0], vec![0.0])
        .unwrap()
        .execute_scalar(0, ctx)
        .unwrap();
    ConstantArray::new(scalar, ROWS).into_array()
}

/// Wrap `array` with seeded random nulls at `density` percent. Zero density stays unwrapped, as a
/// non-nullable column would.
fn with_nulls(array: ArrayRef, seed: u64, density: usize) -> ArrayRef {
    if density == 0 {
        return array;
    }

    let valid = (0..ROWS).map(|i| (splitmix64(seed ^ i as u64) % 100) >= density as u64);
    MaskedArray::try_new(array, Validity::from_iter(valid))
        .unwrap()
        .into_array()
}

/// One arm over the operand pair: `Some` forces a strategy through the harness seam, `None` runs
/// the full pipeline with the per-batch selection.
fn bench_contains(bencher: Bencher, a: ArrayRef, b: ArrayRef, strategy: Option<NullStrategy>) {
    let mut ctx = SESSION.create_execution_ctx();

    bencher
        .counter(ItemsCount::new(ROWS))
        .bench_local(|| match strategy {
            None => GeoContains::try_new_array(a.clone(), b.clone())
                .unwrap()
                .into_array()
                .execute::<Canonical>(&mut ctx)
                .unwrap(),
            Some(strategy) => execute_row_fn_with_strategy(
                &GeoContains,
                &EmptyOptions,
                vec![a.clone(), b.clone()],
                ROWS,
                strategy,
                &mut ctx,
            )
            .unwrap()
            .execute::<Canonical>(&mut ctx)
            .unwrap(),
        });
}

/// Column of polygons CONTAINS constant point, nulls on the polygon column.
mod polygons_x_constant_point {
    use super::*;

    fn operands(density: usize) -> (ArrayRef, ArrayRef) {
        let mut ctx = SESSION.create_execution_ctx();
        (with_nulls(squares(), 1, density), constant_point(&mut ctx))
    }

    #[divan::bench(args = DENSITIES)]
    fn filter(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, Some(NullStrategy::Filter));
    }

    #[divan::bench(args = DENSITIES)]
    fn branch(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, Some(NullStrategy::BranchAndSkip));
    }

    #[divan::bench(args = DENSITIES)]
    fn auto(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, None);
    }
}

/// Column of polygons CONTAINS column of points, independent nulls on both, so the conjoined
/// valid fraction is roughly `(1 - d)^2`.
mod polygons_x_points {
    use super::*;

    fn operands(density: usize) -> (ArrayRef, ArrayRef) {
        (
            with_nulls(squares(), 1, density),
            with_nulls(points(), 2, density),
        )
    }

    #[divan::bench(args = DENSITIES)]
    fn filter(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, Some(NullStrategy::Filter));
    }

    #[divan::bench(args = DENSITIES)]
    fn branch(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, Some(NullStrategy::BranchAndSkip));
    }

    #[divan::bench(args = DENSITIES)]
    fn auto(bencher: Bencher, density: usize) {
        let (a, b) = operands(density);
        bench_contains(bencher, a, b, None);
    }
}
