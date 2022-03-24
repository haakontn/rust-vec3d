//! # Small crate for representing vectors in three dimensions
//! 
//! This crate contains the struct [`Vec3d`] which can be used with
//! both [`f32`] and [`f64`] floating point numbers. The main goal is to 
//! both be able to represent points in 3D space as well as 
//! performing some of the most common mathematical operations 
//! on such vectors for numerical simulations.
//! The storage is done in cartesian coordinates, but spherical
//! coordinates can be used in the creation of a new vector with
//! the function [`Vec3d::from_spherical()`]

use num_traits::Float;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// Cartesian vector in 3 dimensions
///
/// The struct stores a vector in 3D space as collection of
/// three coordinates x, y, and z.
#[derive(Clone, Copy, Debug)]
pub struct Vec3d<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Float> Default for Vec3d<T> {
    /// The zero vector
    fn default() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<T: Float> Vec3d<T> {
    /// Create a new vector based on cartesian coordinates
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    /// Create a new vector based on spherical coordinates
    pub fn from_spherical(r: T, phi: T, theta: T) -> Self {
        Self {
            x: r * theta.cos() * phi.cos(),
            y: r * theta.cos() * phi.sin(),
            z: r * theta.sin(),
        }
    }

    /// Check if any component is not a number
    pub fn is_nan(&self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Geometrical length of Vec3d instance
    pub fn len(&self) -> T {
        self.dot(self).sqrt()
    }

    /// Geometrical length of Vec3d instance
    /// 
    /// Alternative name for the len() function
    pub fn norm2(&self) -> T {
        self.len()
    }

    /// Compute the (smallest) angle between two Vec3d instances
    pub fn angle(&self, rhs: &Vec3d<T>) -> T {
        let dot_prod = self.dot(rhs);
        let cos_theta = dot_prod / (self.len() * rhs.len());

        cos_theta.acos()
    }

    /// Compute the unit normal vector relative two Vec3d instances
    pub fn normal(&self, rhs: &Vec3d<T>) -> Vec3d<T> {
        // Obtain a unit vector normal to the
        // First, get direction
        let n_unscaled = self.cross(rhs);

        // Normalize the length before returning
        n_unscaled / n_unscaled.len()
    }

    /// Perform cross product between two Vec3d instances
    pub fn cross(&self, rhs: &Vec3d<T>) -> Vec3d<T> {
        Vec3d {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }

    /// Perform the dot product between two Vec3d instances
    pub fn dot(&self, rhs: &Vec3d<T>) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

// Summation of Vec<&Vec3d<T>>
impl<'a, T: Float> Sum<&'a Self> for Vec3d<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Vec3d::default(), |total, &x| total + x)
    }
}

// Summation of Vec<Vec3d<T>>
impl<T: Float> Sum<Self> for Vec3d<T> {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Vec3d::default(), |total, x| total + x)
    }
}

// Compare equality with some precision since Vec3d is solely
// based on floating point numbers (at least currently)
impl<T: Float> PartialEq for Vec3d<T> {
    fn eq(&self, rhs: &Self) -> bool {
        // Ensure we are not comparing bits
        // TODO Modify this to take the magnitude into account, but make sure
        // to handle the case of zero values as well... Maybe take the
        // absolute value plus the machine epsilon?
        (self.x - rhs.x).abs() < T::epsilon()
            && (self.y - rhs.y).abs() < T::epsilon()
            && (self.z - rhs.z).abs() < T::epsilon()
    }
}

impl<T: Float> Add for Vec3d<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: AddAssign> AddAssign for Vec3d<T> {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: Sub<Output = T>> Sub for Vec3d<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Vec3d<T> {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// Enable multiplication with a scalar (Float)
impl<T: Float + Mul<T, Output = T>> Mul<T> for Vec3d<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self {
        Vec3d {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Float + MulAssign> MulAssign<T> for Vec3d<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

// Make the multiplication commutable
impl Mul<Vec3d<f32>> for f32 {
    type Output = Vec3d<f32>;

    fn mul(self, rhs: Vec3d<f32>) -> Self::Output {
        Vec3d {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Mul<Vec3d<f64>> for f64 {
    type Output = Vec3d<f64>;

    fn mul(self, rhs: Vec3d<f64>) -> Self::Output {
        Vec3d {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

// Enable division with a scalar (Float)
// Note this is not commutative!
impl<T: Float> Div<T> for Vec3d<T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Vec3d {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    type Float = f64;

    #[test]
    fn comparison() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 4.0);
        let b: Vec3d<Float> = Vec3d::new(1.0, 2.0, 4.0);

        assert_eq!(a, b);
    }

    #[test]
    fn cross() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 6.0);
        let b: Vec3d<Float> = Vec3d::new(7.0, 3.0, 0.5);

        let a_cross_b = Vec3d::new(-17.0, 41.5, -11.0);

        assert_eq!(a.cross(&b), a_cross_b);
        assert_eq!(b.cross(&a), -a_cross_b);
    }

    #[test]
    fn dot() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);
        let b: Vec3d<Float> = Vec3d::new(2.5, 3.0, 0.20);

        assert!((a.dot(&b) - (2.5 + 6.0 + 1.0)).abs() < Float::EPSILON);
    }

    #[test]
    fn len() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);

        assert!((a.len() - Float::sqrt(1.0 + 4.0 + 25.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn angle_xy() {
        // Run simple 2D test
        let angle: Float = 0.35;
        let a: Vec3d<Float> = Vec3d::new(1.0, 0.0, 0.0);
        let b: Vec3d<Float> = Vec3d::new(angle.cos(), angle.sin(), 0.0);

        assert!((a.angle(&b) - angle).abs() < Float::EPSILON);
    }
    #[test]
    fn angle_xz() {
        // Run simple 2D test
        let angle: Float = 0.35;
        let a: Vec3d<Float> = Vec3d::new(1.0, 0.0, 0.0);
        let b: Vec3d<Float> = Vec3d::new(angle.cos(), 0.0, angle.sin());

        assert!((a.angle(&b) - angle).abs() < Float::EPSILON);
    }

    #[test]
    fn normal() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 0.0, 0.0);
        let b: Vec3d<Float> = Vec3d::new(0.0, 1.0, 0.0);

        assert_eq!(a.normal(&b), Vec3d::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn add() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);
        let b: Vec3d<Float> = Vec3d::new(2.0, 0.0, -1.0);

        assert_eq!(a + b, Vec3d::new(3.0, 2.0, 4.0));
    }

    #[test]
    fn add_assign() {
        let mut a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);
        let b: Vec3d<Float> = Vec3d::new(2.0, 0.0, -1.0);
        a += b;

        assert_eq!(a, Vec3d::new(3.0, 2.0, 4.0));
    }

    #[test]
    fn sub() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);
        let b: Vec3d<Float> = Vec3d::new(2.0, 0.0, -1.0);

        assert_eq!(a - b, Vec3d::new(-1.0, 2.0, 6.0));
    }

    #[test]
    fn neg() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);

        assert_eq!(-a, Vec3d::new(-1.0, -2.0, -5.0));
    }

    #[test]
    fn mul() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);

        assert_eq!(2.0 * a, Vec3d::new(2.0, 4.0, 10.0));
    }

    #[test]
    fn mul_commutative() {
        // Check that the multiplication with a float is commutative
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);

        assert_eq!(0.7 * a, a * 0.7);
    }

    #[test]
    fn div() {
        let a: Vec3d<Float> = Vec3d::new(1.0, 2.0, 5.0);

        assert_eq!(a / 2.0, Vec3d::new(0.5, 1.0, 2.5));
    }

    #[test]
    fn sum() {
        let n_vecs = 13;
        let unit_vec: Vec3d<Float> = Vec3d::new(1.0, 1.5, 0.75);
        let unit_vecs = vec![unit_vec; n_vecs];
        let iterator_sum: Vec3d<Float> = unit_vecs.iter().sum();

        assert_eq!(iterator_sum / (n_vecs as Float), unit_vec)
    }
}
