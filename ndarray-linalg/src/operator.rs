//! Linear operator algebra

use crate::generate::hstack;
use crate::types::*;
use ndarray::*;

/// Abstracted linear operator as an action to vector (`ArrayBase<S, Ix1>`) and matrix
/// (`ArrayBase<S, Ix2`)
pub trait LinearOperator {
    type Elem: Scalar;

    /// Apply operator out-place
    fn apply(&self, a: &ArrayRef<Self::Elem, Ix1>) -> Array1<Self::Elem> {
        let mut a = a.to_owned();
        self.apply_mut(&mut a);
        a
    }

    /// Apply operator in-place
    fn apply_mut(&self, a: &mut ArrayRef<Self::Elem, Ix1>) {
        let b = self.apply(a);
        azip!((a in a, &b in &b) *a = b);
    }

    /// Apply operator with move
    fn apply_into<S>(&self, mut a: ArrayBase<S, Ix1>) -> ArrayBase<S, Ix1>
    where
        S: DataOwned<Elem = Self::Elem> + DataMut,
    {
        self.apply_mut(&mut a);
        a
    }

    /// Apply operator to matrix out-place
    fn apply2(&self, a: &ArrayRef<Self::Elem, Ix2>) -> Array2<Self::Elem> {
        let cols: Vec<_> = a.axis_iter(Axis(1)).map(|col| self.apply(&col)).collect();
        hstack(&cols).unwrap()
    }

    /// Apply operator to matrix in-place
    fn apply2_mut(&self, a: &mut ArrayRef<Self::Elem, Ix2>) {
        for mut col in a.axis_iter_mut(Axis(1)) {
            self.apply_mut(&mut col)
        }
    }

    /// Apply operator to matrix with move
    fn apply2_into<S>(&self, mut a: ArrayBase<S, Ix2>) -> ArrayBase<S, Ix2>
    where
        S: DataOwned<Elem = Self::Elem> + DataMut,
    {
        self.apply2_mut(&mut a);
        a
    }
}

impl<A, Sa> LinearOperator for ArrayBase<Sa, Ix2>
where
    A: Scalar,
    Sa: Data<Elem = A>,
{
    type Elem = A;

    fn apply(&self, a: &ArrayRef<A, Ix1>) -> Array1<A> {
        self.dot(a)
    }
}
