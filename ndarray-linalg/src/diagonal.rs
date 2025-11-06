//! Vector as a Diagonal matrix

use ndarray::*;

use super::operator::*;
use super::types::*;

/// Vector as a Diagonal matrix
pub struct Diagonal<S: Data> {
    diag: ArrayBase<S, Ix1>,
}

pub trait IntoDiagonal<S: Data> {
    fn into_diagonal(self) -> Diagonal<S>;
}

pub trait AsDiagonal<A> {
    fn as_diagonal(&self) -> Diagonal<ViewRepr<&A>>;
}

impl<S: Data> IntoDiagonal<S> for ArrayBase<S, Ix1> {
    fn into_diagonal(self) -> Diagonal<S> {
        Diagonal { diag: self }
    }
}

impl<A> AsDiagonal<A> for ArrayRef<A, Ix1> {
    fn as_diagonal(&self) -> Diagonal<ViewRepr<&A>> {
        Diagonal { diag: self.view() }
    }
}

impl<A, Sa> LinearOperator for Diagonal<Sa>
where
    A: Scalar,
    Sa: Data<Elem = A>,
{
    type Elem = A;

    fn apply_mut(&self, a: &mut ArrayRef<A, Ix1>) {
        for (val, d) in a.iter_mut().zip(self.diag.iter()) {
            *val *= *d;
        }
    }
}
