use crate::types::*;
use ndarray::*;

/// Inner Product
///
/// Differenct from `Dot` trait, this take complex conjugate of `self` elements
///
pub trait InnerProduct {
    type Elem: Scalar;

    /// Inner product `(self.conjugate, rhs)
    fn inner(&self, rhs: &ArrayRef<Self::Elem, Ix1>) -> Self::Elem;
}

impl<A> InnerProduct for ArrayRef<A, Ix1>
where
    A: Scalar,
{
    type Elem = A;

    fn inner(&self, rhs: &ArrayRef<A, Ix1>) -> A {
        assert_eq!(self.len(), rhs.len());
        Zip::from(self)
            .and(rhs)
            .fold_while(A::zero(), |acc, s, r| {
                FoldWhile::Continue(acc + s.conj() * *r)
            })
            .into_inner()
    }
}
