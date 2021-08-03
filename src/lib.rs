use nalgebra::{
    allocator::{Allocator, SameShapeAllocator},
    constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint},
    storage::Storage,
    Const, DefaultAllocator, Dim, DimMin, DimMinimum, DimSub, Matrix, OMatrix,
};

fn orthogonal_decomposition<D: Dim, N: Dim, S1: Storage<f64, D, D>, S2: Storage<f64, D, N>>(
    column_space: Matrix<f64, D, D, S1>,
    x: Matrix<f64, D, N, S2>,
) -> (
    Matrix<f64, D, N, <DefaultAllocator as nalgebra::allocator::Allocator<f64, D, N>>::Buffer>,
    Matrix<
        f64,
        <ShapeConstraint as SameNumberOfRows<D, D>>::Representative,
        <ShapeConstraint as SameNumberOfColumns<N, N>>::Representative,
        <DefaultAllocator as nalgebra::allocator::Allocator<
            f64,
            <ShapeConstraint as SameNumberOfRows<D, D>>::Representative,
            <ShapeConstraint as SameNumberOfColumns<N, N>>::Representative,
        >>::Buffer,
    >,
)
where
    S1: Clone,
    S2: Clone,
    D: DimMin<D>,
    DefaultAllocator: Allocator<f64, D, D>
        + Allocator<f64, D, N>
        + Allocator<f64, DimMinimum<D, D>, N>
        + Allocator<f64, <D as DimMin<D>>::Output>
        + Allocator<f64, D, <D as DimMin<D>>::Output>
        + Allocator<f64, <D as DimMin<D>>::Output, D>
        + Allocator<f64, D>
        + Allocator<f64, <<D as DimMin<D>>::Output as DimSub<Const<1_usize>>>::Output>
        + SameShapeAllocator<f64, D, N, D, N>,
    ShapeConstraint: SameNumberOfRows<D, D> + SameNumberOfColumns<N, N>,
    <D as DimMin<D>>::Output: DimSub<Const<1_usize>>,
{
    let column_space_t = column_space.transpose();
    let symmetric = (column_space_t.clone() * column_space.clone()).svd(true, true);
    let column_space_tx = column_space_t * x.clone();
    let sol: OMatrix<f64, D, N> = symmetric.solve(&column_space_tx, 1e-10).unwrap();
    let column_space_component = column_space * sol;
    let orthogonal_component = x - &column_space_component;
    (column_space_component, orthogonal_component)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Matrix3x4, Vector3};

    #[test]
    fn basic_single_vector_decomposition() {
        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 1.);
        let x = Vector3::new(3.71, 105.12, -168.7);

        let decomposition = orthogonal_decomposition(a, x);
        let expected = (Vector3::new(3.71, 0., -168.7), Vector3::new(0., 105.12, 0.));

        assert_eq!(decomposition, expected);
    }

    #[test]
    fn basic_multiple_vectors_as_matrix_decompositions() {
        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 1.);
        let x1 = Vector3::new(2.3, 4.3, 7.6);
        let x2 = Vector3::new(8.4, 5.9, 2.9);
        let x3 = Vector3::new(9.4, 6.8, 1.2);
        let x4 = Vector3::new(3.5, 8.3, 4.4);
        let x = Matrix3x4::from_columns(&[x1, x2, x3, x4]);

        let decomposition = orthogonal_decomposition(a, x);
        let expected = (
            Matrix3x4::from_columns(&[
                Vector3::new(2.3, 0., 7.6),
                Vector3::new(8.4, 0., 2.9),
                Vector3::new(9.4, 0., 1.2),
                Vector3::new(3.5, 0., 4.4),
            ]),
            Matrix3x4::from_columns(&[
                Vector3::new(0., 4.3, 0.),
                Vector3::new(0., 5.9, 0.),
                Vector3::new(0., 6.8, 0.),
                Vector3::new(0., 8.3, 0.),
            ]),
        );

        assert_eq!(decomposition, expected);
    }
}
