use nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, DimMin, DimName, DimSub, Matrix,
    OMatrix, U1,
};

#[derive(Clone, Debug)]
pub struct SubSpace<D: Dim>
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    pub column_space: OMatrix<f64, D, D>,
    pub epsilon: f64,
}

impl<D: Dim> SubSpace<D>
where
    DefaultAllocator: Allocator<f64, D, D>,
{
    fn projection<N, S1>(&self, x: &Matrix<f64, D, N, S1>) -> OMatrix<f64, D, N>
    where
        S1: Clone + Storage<f64, D, N>,
        D: DimMin<D, Output = D> + DimSub<U1>,
        N: Dim,
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, D, N>
            + Allocator<f64, D>
            + Allocator<f64, <D as DimSub<U1>>::Output>,
    {
        let column_space_t = self.column_space.transpose();
        let symmetric = (column_space_t.clone() * self.column_space.clone()).svd(true, true);
        let column_space_tx = column_space_t * x.clone();
        let sol = symmetric.solve(&column_space_tx, self.epsilon).unwrap();
        let projection: OMatrix<f64, D, N> = &self.column_space * sol;
        projection
    }

    fn orthogonal_projection<N, S1>(&self, x: &Matrix<f64, D, N, S1>) -> OMatrix<f64, D, N>
    where
        S1: Clone + Storage<f64, D, N>,
        D: DimMin<D, Output = D> + DimSub<U1>,
        N: Dim,
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, D, N>
            + Allocator<f64, D>
            + Allocator<f64, <D as DimSub<U1>>::Output>,
    {
        let projection = self.projection(x);
        x - projection
    }

    fn orthogonal_decomposition<N, S1>(
        &self,
        x: &Matrix<f64, D, N, S1>,
    ) -> (OMatrix<f64, D, N>, OMatrix<f64, D, N>)
    where
        S1: Clone + Storage<f64, D, N>,
        D: DimMin<D, Output = D> + DimSub<U1>,
        N: Dim,
        DefaultAllocator: Allocator<f64, D, D>
            + Allocator<f64, D, N>
            + Allocator<f64, D>
            + Allocator<f64, <D as DimSub<U1>>::Output>,
    {
        let projection = self.projection(x);
        let orthogonal_projection = x - &projection;
        (projection, orthogonal_projection)
    }

    fn span(
        vs: &mut [Matrix<f64, D, U1, <DefaultAllocator as Allocator<f64, D, U1>>::Buffer>],
        epsilon: f64,
    ) -> SubSpace<D>
    where
        D: DimName,
        DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
    {
        let mut column_space = OMatrix::<f64, D, D>::from_element(0.);
        let n_free_vectors = Matrix::orthonormalize(vs);
        for i in 0..n_free_vectors {
            column_space.set_column(i, &vs[i]);
        }
        SubSpace {
            column_space,
            epsilon,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Matrix3x4, Vector3};

    const EPSILON: f64 = 1e-10;

    #[test]
    fn basic_single_vector_decomposition() {
        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 1.);
        let x = Vector3::new(3.71, 105.12, -168.7);

        let sub_space = SubSpace {
            column_space: a,
            epsilon: EPSILON,
        };
        let decomposition = sub_space.orthogonal_decomposition(&x);
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

        let sub_space = SubSpace {
            column_space: a,
            epsilon: EPSILON,
        };
        let decomposition = sub_space.orthogonal_decomposition(&x);
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

    #[test]
    fn span_tests() {
        let spanned_sub_space = SubSpace::span(
            &mut [
                Vector3::new(0., 0., -6.5),
                Vector3::new(1., 2., -23.),
                Vector3::new(1., 2., 0.),
                Vector3::new(0., 0., 1.),
                Vector3::new(2., 4., 0.),
            ],
            EPSILON,
        );

        let a = Matrix3::new(-3.798, 4., 0., 2. * -3.798, 8., 0., 0., 0., 2.);
        let a_sub_space = SubSpace {
            column_space: a,
            epsilon: EPSILON,
        };

        let x1 = Vector3::new(2.3, 4.3, 7.6);
        let x2 = Vector3::new(8.4, 5.9, 2.9);
        let x3 = Vector3::new(9.4, 6.8, 1.2);
        let x4 = Vector3::new(3.5, 8.3, 4.4);
        let x = Matrix3x4::from_columns(&[x1, x2, x3, x4]);

        let from_spanned = spanned_sub_space.orthogonal_decomposition(&x);
        let from_matrix = a_sub_space.orthogonal_decomposition(&x);

        assert!((&from_spanned.0 - &from_matrix.0).norm() < EPSILON);
        assert!((&from_spanned.1 - &from_matrix.1).norm() < EPSILON);
    }
}
