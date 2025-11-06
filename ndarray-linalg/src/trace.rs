//! Trace calculation

use ndarray::*;
use std::iter::Sum;

use super::error::*;
use super::types::*;

pub trait Trace {
    type Output;
    fn trace(&self) -> Result<Self::Output>;
}

impl<A> Trace for ArrayRef<A, Ix2>
where
    A: Scalar + Sum,
{
    type Output = A;

    fn trace(&self) -> Result<Self::Output> {
        let n = match self.is_square() {
            true => Ok(self.nrows()),
            false => Err(LinalgError::NotSquare {
                rows: self.nrows() as i32,
                cols: self.ncols() as i32,
            }),
        }?;
        Ok((0..n as usize).map(|i| self[(i, i)]).sum())
    }
}
