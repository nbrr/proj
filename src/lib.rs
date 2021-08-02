use nalgebra::allocator::{Allocator, SameShapeAllocator};
use nalgebra::constraint::{SameNumberOfColumns, SameNumberOfRows, ShapeConstraint};
use nalgebra::{
    storage::Storage, ComplexField, DefaultAllocator, Dim, DimDiff, DimMin, DimMinimum, DimSub,
    Matrix, Vector, U1,
};

fn proj<D: Dim, S1: Storage<f64, D, D>, S2: Storage<f64, D, U1>>(
    colspace: Matrix<f64, D, D, S1>,
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
        + Allocator<f64, D>
        + Allocator<f64, DimDiff<DimMinimum<D, D>, U1>>
        + Allocator<f64, DimMinimum<D, D>, D>
        + Allocator<f64, D, DimMinimum<D, D>>
        + Allocator<f64, DimMinimum<D, D>>
        + Allocator<<f64 as ComplexField>::RealField, DimMinimum<D, D>>
        + Allocator<<f64 as ComplexField>::RealField, DimDiff<DimMinimum<D, D>, U1>>,
    S2: Storage<f64, D, U1>,
    DefaultAllocator: Allocator<f64, D, U1> + Allocator<f64, DimMinimum<D, D>, U1>,
    ShapeConstraint: SameNumberOfRows<D, D>,
    DefaultAllocator: SameShapeAllocator<f64, D, D, D, D>,
    ShapeConstraint: SameNumberOfRows<D, D> + SameNumberOfColumns<D, D>,
    DefaultAllocator: Allocator<f64, <ShapeConstraint as SameNumberOfRows<D, D>>::Representative>,
{
    let colspace_t = colspace.transpose();
    let sym = (colspace_t.clone() * colspace.clone()).svd(true, true);
    let colspace_tx = colspace_t * x.clone();
    let proj_colspace = colspace * sym.solve(&colspace_tx, 1e-10).unwrap();
    // let proj_orth = x - &proj_colspace;
    (proj_colspace.clone(), x - &proj_colspace)
}

#[cfg(test)]
mod tests {
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn it_works() {
        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 0.);
        let at = a.transpose();

        let x = Vector3::new(3.71, 105.12, -168.7);

        let atx = &at * x;

        let ata_svd = (&at * a).svd(true, true);

        let sol = ata_svd.solve(&atx, 1e-10);

        println!("{:?}", sol);
    }

    #[test]
    fn it_works2() {
        use super::*;

        let a = Matrix3::new(1., 0., 0., 0., 0., 0., 0., 0., 0.);
        let x = Vector3::new(3.71, 105.12, -168.7);

        let sol = proj(a, x);

        println!("{:?}", sol);
    }
}
