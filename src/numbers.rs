use ordered_float::OrderedFloat;

/// Things that can be assigned a positive number.
///
/// Should start from 0. It's used to assign numbers to Action in the Levin search.
pub trait Enumerable {
    fn to_number(&self) -> usize;
}

impl Enumerable for () {
    #[inline]
    fn to_number(&self) -> usize {
        0
    }
}

impl Enumerable for bool {
    #[inline]
    fn to_number(&self) -> usize {
        if *self {
            1
        } else {
            0
        }
    }
}

impl Enumerable for usize {
    #[inline]
    fn to_number(&self) -> usize {
        *self
    }
}

impl Enumerable for u64 {
    #[inline]
    fn to_number(&self) -> usize {
        *self as usize
    }
}

/// The kind of numbers you can use with Levin search.
pub trait ContextModelNumber: Clone + Ord {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(v: usize) -> Self;
    fn from_f64(v: f64) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn sqrt(&self) -> Self;

    fn recip(&self) -> Self {
        let mut one = Self::one();
        one.inplace_div(self);
        one
    }

    fn inplace_mul(&mut self, other: &Self);
    fn inplace_add(&mut self, other: &Self);
    fn inplace_sub(&mut self, other: &Self);
    fn inplace_div(&mut self, other: &Self);
}

impl ContextModelNumber for OrderedFloat<f64> {
    #[inline]
    fn zero() -> Self {
        OrderedFloat(0.0)
    }

    #[inline]
    fn one() -> Self {
        OrderedFloat(1.0)
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        OrderedFloat(v as f64)
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        OrderedFloat(v)
    }

    #[inline]
    fn exp(&self) -> Self {
        OrderedFloat(self.0.exp())
    }

    #[inline]
    fn ln(&self) -> Self {
        OrderedFloat(self.0.ln())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        OrderedFloat(self.0.sqrt())
    }

    #[inline]
    fn inplace_mul(&mut self, other: &Self) {
        self.0 *= other.0;
    }

    #[inline]
    fn inplace_add(&mut self, other: &Self) {
        self.0 += other.0;
    }

    #[inline]
    fn inplace_sub(&mut self, other: &Self) {
        self.0 -= other.0;
    }

    #[inline]
    fn inplace_div(&mut self, other: &Self) {
        self.0 /= other.0;
    }
}

impl ContextModelNumber for OrderedFloat<f32> {
    #[inline]
    fn zero() -> Self {
        OrderedFloat(0.0_f32)
    }

    #[inline]
    fn one() -> Self {
        OrderedFloat(1.0_f32)
    }

    #[inline]
    fn from_usize(v: usize) -> Self {
        OrderedFloat(v as f32)
    }

    #[inline]
    fn from_f64(v: f64) -> Self {
        OrderedFloat(v as f32)
    }

    #[inline]
    fn exp(&self) -> Self {
        OrderedFloat(self.0.exp())
    }

    #[inline]
    fn ln(&self) -> Self {
        OrderedFloat(self.0.ln())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        OrderedFloat(self.0.sqrt())
    }

    #[inline]
    fn inplace_mul(&mut self, other: &Self) {
        self.0 *= other.0;
    }

    #[inline]
    fn inplace_add(&mut self, other: &Self) {
        self.0 += other.0;
    }

    #[inline]
    fn inplace_sub(&mut self, other: &Self) {
        self.0 -= other.0;
    }

    #[inline]
    fn inplace_div(&mut self, other: &Self) {
        self.0 /= other.0;
    }
}
