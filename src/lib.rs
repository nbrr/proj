use nalgebra::{
    allocator::{Allocator, SameShapeAllocator},
    constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    storage::Storage,
    ComplexField, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimSub, Matrix, Vector, U1,
};

fn orthogonal_decomposition<D: Dim, S1: Storage<f64, D, D>, S2: Storage<f64, D, U1>>(
    column_space: Matrix<f64, D, D, S1>,
    x: Vector<f64, D, S2>,
) -> (
    Vector<f64, D, <DefaultAllocator as nalgebra::allocator::Allocator<f64, D>>::Buffer>,
    Vector<
        f64,
        <ShapeConstraint as SameNumberOfRows<D, D>>::Representative,
        <DefaultAllocator as nalgebra::allocator::Allocator<
            f64,
            <ShapeConstraint as SameNumberOfRows<D, D>>::Representative,
        >>::Buffer,
    >,
)
where
    S1: Clone,
    S2: Clone,
    D: DimMin<D>,
    DimMinimum<D, D>: DimSub<U1>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D>
        + Allocator<f64, DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<f64, DimMinimum<D, D>, D>
        + Allocator<f64, D, DimMinimum<D, D>>
        + Allocator<f64, DimMinimum<D, D>>
        + Allocator<<f64 as ComplexField>::RealField, DimMinimum<D, D>>
        + Allocator<<f64 as ComplexField>::RealField, DimDiff<DimMinimum<D, D>, U1>>,
    S2: Storage<f64, D, U1>,
{
    let column_space_t = column_space.transpose();
    let symmetric = (column_space_t.clone() * column_space.clone()).svd(true, true);
    let column_space_tx = column_space_t * x.clone();
    let column_space_component = column_space * symmetric.solve(&column_space_tx, 1e-10).unwrap();
    let orthogonal_component = x - &column_space_component;
    (column_space_component, orthogonal_component)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn basic_decomposition() {
        use super::*;

        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 1.);
        let x = Vector3::new(3.71, 105.12, -168.7);

        let decomposition = orthogonal_decomposition(a, x);
        let expected = (Vector3::new(3.71, 0., -168.7), Vector3::new(0., 105.12, 0.));

        assert_eq!(decomposition, expected);
    }
}
