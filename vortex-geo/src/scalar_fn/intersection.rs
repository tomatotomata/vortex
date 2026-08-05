// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Intersection`: pairwise planar intersection of native polygons.

use geo::BooleanOps;
use geo_types::Geometry;
use geo_types::MultiPolygon as GeoMultiPolygon;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::expr::Expression;
use vortex_array::expr::union_child_validities;
use vortex_array::scalar_fn::Arity;
use vortex_array::scalar_fn::ChildName;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ExecutionArgs;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::TypedScalarFnInstance;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::AllOr;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::extension::GeoMetadata;
use crate::extension::MultiPolygon;
use crate::extension::Polygon;
use crate::extension::build_multipolygon_array;
use crate::extension::coordinate::Dimension;
use crate::extension::geometries;
use crate::extension::multipolygon_storage_dtype;
use crate::extension::single_geometry;
use crate::scalar_fn::execute::Execution;
use crate::scalar_fn::execute::Operand;
use crate::scalar_fn::execute::dispatch_binary;

/// Resolve CRS metadata shared by two polygon operands.
fn intersection_metadata(left: &GeoMetadata, right: &GeoMetadata) -> VortexResult<GeoMetadata> {
    match (&left.crs, &right.crs) {
        (Some(left_crs), Some(right_crs)) => {
            vortex_ensure!(
                left_crs == right_crs,
                "geo: intersection operands have different coordinate reference systems: \
                 {left_crs} and {right_crs}"
            );
            Ok(left.clone())
        }
        (Some(_), None) => Ok(left.clone()),
        (None, Some(_)) => Ok(right.clone()),
        (None, None) => Ok(GeoMetadata::default()),
    }
}

/// Resolve the strict native `Polygon x Polygon -> MultiPolygon` overload.
fn intersection_dtype(dtypes: &[DType]) -> VortexResult<ExtDType<MultiPolygon>> {
    vortex_ensure!(
        dtypes.len() == 2,
        "geo: intersection requires exactly two Polygon operands, got {}",
        dtypes.len()
    );
    for dtype in dtypes {
        vortex_ensure!(
            dtype
                .as_extension_opt()
                .is_some_and(|extension| extension.is::<Polygon>()),
            "geo: intersection operand {dtype} is not a native Polygon"
        );
    }

    let left = dtypes[0].as_extension();
    let right = dtypes[1].as_extension();
    let metadata = intersection_metadata(left.metadata::<Polygon>(), right.metadata::<Polygon>())?;
    let nullability = Nullability::from(dtypes.iter().any(DType::is_nullable));
    ExtDType::try_new(
        metadata,
        multipolygon_storage_dtype(Dimension::Xy, nullability),
    )
}

/// Intersect two decoded polygon values.
fn intersect(left: &Geometry<f64>, right: &Geometry<f64>) -> GeoMultiPolygon<f64> {
    let (Geometry::Polygon(left), Geometry::Polygon(right)) = (left, right) else {
        unreachable!("intersection operands were validated as Polygon")
    };
    left.intersection(right)
}

/// Scatter valid intersection results and build their native MultiPolygon array.
fn build_intersections(
    intersections: Vec<GeoMultiPolygon<f64>>,
    execution: &Execution<2>,
    output_dtype: &ExtDType<MultiPolygon>,
) -> VortexResult<ArrayRef> {
    let intersections = match execution.valid.indices() {
        AllOr::All => intersections.into_iter().map(Some).collect(),
        AllOr::None => vec![None; execution.len],
        AllOr::Some(rows) => {
            let mut output = vec![None; execution.len];
            for (&row, intersection) in rows.iter().zip(intersections) {
                output[row] = Some(intersection);
            }
            output
        }
    };
    build_multipolygon_array(
        &intersections,
        output_dtype.metadata().clone(),
        execution.nullability,
    )
}

/// Execute intersection after shared binary shape and null dispatch.
fn execute_intersection(
    execution: Execution<2>,
    output_dtype: &ExtDType<MultiPolygon>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let intersections = match &execution.operands {
        [Operand::Constant(left), Operand::Constant(right)] => {
            let intersection =
                intersect(&single_geometry(left, ctx)?, &single_geometry(right, ctx)?);
            let one = build_multipolygon_array(
                &[Some(intersection)],
                output_dtype.metadata().clone(),
                execution.nullability,
            )?;
            return Ok(ConstantArray::new(one.execute_scalar(0, ctx)?, execution.len).into_array());
        }
        [Operand::Constant(left), Operand::Column(right)] => {
            let left = single_geometry(left, ctx)?;
            geometries(&right.filter(execution.valid.clone())?, ctx)?
                .iter()
                .map(|right| intersect(&left, right))
                .collect()
        }
        [Operand::Column(left), Operand::Constant(right)] => {
            let right = single_geometry(right, ctx)?;
            geometries(&left.filter(execution.valid.clone())?, ctx)?
                .iter()
                .map(|left| intersect(left, &right))
                .collect()
        }
        [Operand::Column(left), Operand::Column(right)] => {
            let left = geometries(&left.filter(execution.valid.clone())?, ctx)?;
            let right = geometries(&right.filter(execution.valid.clone())?, ctx)?;
            left.iter()
                .zip(&right)
                .map(|(left, right)| intersect(left, right))
                .collect()
        }
    };
    build_intersections(intersections, &execution, output_dtype)
}

/// Compute the pairwise two-dimensional intersection of native `Polygon` operands as a native
/// `MultiPolygon`. Disjoint and boundary-only intersections produce an empty `MultiPolygon`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct GeoIntersection;

impl GeoIntersection {
    /// A lazy `ScalarFnArray` intersecting two native polygon operands by row.
    pub fn try_new_array(left: ArrayRef, right: ArrayRef) -> VortexResult<ScalarFnArray> {
        ScalarFnArray::try_new(
            TypedScalarFnInstance::new(GeoIntersection, EmptyOptions).erased(),
            vec![left, right],
        )
    }
}

impl ScalarFnVTable for GeoIntersection {
    type Options = EmptyOptions;

    fn id(&self) -> ScalarFnId {
        static ID: CachedId = CachedId::new("vortex.geo.intersection");
        *ID
    }

    fn serialize(&self, _: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(&self, _: &[u8], _: &VortexSession) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn arity(&self, _: &Self::Options) -> Arity {
        Arity::Exact(2)
    }

    fn child_name(&self, _: &Self::Options, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("left"),
            1 => ChildName::from("right"),
            _ => unreachable!("intersection has exactly two children"),
        }
    }

    fn return_dtype(&self, _: &Self::Options, dtypes: &[DType]) -> VortexResult<DType> {
        Ok(DType::Extension(intersection_dtype(dtypes)?.erased()))
    }

    fn execute(
        &self,
        _: &Self::Options,
        args: &dyn ExecutionArgs,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let left = args.get(0)?;
        let right = args.get(1)?;
        let output_dtype = intersection_dtype(&[left.dtype().clone(), right.dtype().clone()])?;
        dispatch_binary(
            &left,
            &right,
            DType::Extension(output_dtype.clone().erased()),
            |execution, ctx| execute_intersection(execution, &output_dtype, ctx),
            ctx,
        )
    }

    fn validity(
        &self,
        _: &Self::Options,
        expression: &Expression,
    ) -> VortexResult<Option<Expression>> {
        union_child_validities(expression)
    }

    fn is_strict(&self, _: &Self::Options) -> bool {
        true
    }

    fn is_fallible(&self, _: &Self::Options) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use geo_types::Geometry;
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::Columnar;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::MaskedArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::GeoIntersection;
    use crate::extension::MultiPolygon;
    use crate::extension::geometries;
    use crate::scalar_fn::area::GeoArea;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;

    fn square(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Vec<(f64, f64)> {
        vec![
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
    }

    fn polygon_constant(
        ring: Vec<(f64, f64)>,
        len: usize,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let scalar = polygon_column(vec![vec![ring]])?.execute_scalar(0, ctx)?;
        Ok(ConstantArray::new(scalar, len).into_array())
    }

    #[test]
    fn q9_area_pipeline_handles_overlap_disjoint_and_touching() -> VortexResult<()> {
        let left = polygon_column(vec![
            vec![square(0.0, 0.0, 2.0, 2.0)],
            vec![square(0.0, 0.0, 1.0, 1.0)],
            vec![square(0.0, 0.0, 1.0, 1.0)],
        ])?;
        let right = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(2.0, 2.0, 3.0, 3.0)],
            vec![square(1.0, 0.0, 2.0, 1.0)],
        ])?;
        let intersections = GeoIntersection::try_new_array(left, right)?.into_array();
        assert!(intersections.dtype().as_extension().is::<MultiPolygon>());

        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let decoded = geometries(&intersections, &mut ctx)?;
        let polygon_counts = decoded
            .iter()
            .map(|geometry| match geometry {
                Geometry::MultiPolygon(multipolygon) => Ok(multipolygon.0.len()),
                other => Err(vortex_err!(
                    "intersection decoded as {other:?}, expected MultiPolygon"
                )),
            })
            .collect::<VortexResult<Vec<_>>>()?;
        assert_eq!(polygon_counts, [1, 0, 0]);

        let areas = GeoArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([1.0_f64, 0.0, 0.0]).into_array();
        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn preserves_holes() -> VortexResult<()> {
        let left = polygon_column(vec![vec![
            square(0.0, 0.0, 4.0, 4.0),
            square(1.0, 1.0, 3.0, 3.0),
        ]])?;
        let right = polygon_column(vec![vec![square(2.0, 0.0, 5.0, 4.0)]])?;
        let intersections = GeoIntersection::try_new_array(left, right)?.into_array();
        let areas = GeoArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([6.0_f64]).into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn propagates_nulls() -> VortexResult<()> {
        let left = MaskedArray::try_new(
            polygon_column(vec![
                vec![square(0.0, 0.0, 2.0, 2.0)],
                vec![square(0.0, 0.0, 2.0, 2.0)],
            ])?,
            Validity::from_iter([true, false]),
        )?
        .into_array();
        let right = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(1.0, 1.0, 3.0, 3.0)],
        ])?;
        let intersections = GeoIntersection::try_new_array(left, right)?.into_array();
        let areas = GeoArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::new(vec![1.0_f64, 0.0], Validity::from_iter([true, false]))
            .into_array();
        let mut ctx = vortex_array::array_session().create_execution_ctx();

        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case::constant_left(true)]
    #[case::constant_right(false)]
    fn pairs_constants_with_columns(#[case] constant_left: bool) -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let constant = polygon_constant(square(0.0, 0.0, 2.0, 2.0), 2, &mut ctx)?;
        let column = polygon_column(vec![
            vec![square(1.0, 1.0, 3.0, 3.0)],
            vec![square(3.0, 3.0, 4.0, 4.0)],
        ])?;
        let (left, right) = if constant_left {
            (constant, column)
        } else {
            (column, constant)
        };

        let intersections = GeoIntersection::try_new_array(left, right)?.into_array();
        let areas = GeoArea::try_new_array(intersections)?.into_array();
        let expected = PrimitiveArray::from_iter([1.0_f64, 0.0]).into_array();
        assert_arrays_eq!(areas, expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn two_constants_remain_constant() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let left = polygon_constant(square(0.0, 0.0, 2.0, 2.0), 3, &mut ctx)?;
        let right = polygon_constant(square(1.0, 1.0, 3.0, 3.0), 3, &mut ctx)?;

        let result = GeoIntersection::try_new_array(left, right)?.into_array();
        let Columnar::Constant(constant) = result.execute::<Columnar>(&mut ctx)? else {
            return Err(vortex_err!(
                "intersection of two constants should remain constant"
            ));
        };
        assert_eq!(constant.len(), 3);
        Ok(())
    }

    #[rstest]
    #[case::none(0)]
    #[case::one(1)]
    #[case::three(3)]
    fn rejects_wrong_arity(#[case] arity: usize) -> VortexResult<()> {
        let dtype = polygon_column(vec![vec![]])?.dtype().clone();
        assert!(
            GeoIntersection
                .return_dtype(&EmptyOptions, &vec![dtype; arity])
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn rejects_non_polygon_input() -> VortexResult<()> {
        let polygon = polygon_column(vec![vec![]])?;
        let point = point_column(vec![0.0], vec![0.0])?;
        assert!(GeoIntersection::try_new_array(polygon, point).is_err());
        Ok(())
    }
}
