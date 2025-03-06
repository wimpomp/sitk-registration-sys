mod sys;

use crate::sys::{interp, register};
use anyhow::{Result, anyhow};
use ndarray::{Array2, ArrayView2, AsArray, Ix2, array, s};
use serde::{Deserialize, Serialize};
use serde_yaml::{from_reader, to_writer};
use std::fs::File;
use std::ops::Mul;
use std::path::PathBuf;

/// a trait marking number types that can be used in sitk:
/// (u/i)(8/16/32/64), (u/i)size, f(32/64)
pub trait PixelType: Clone {
    const PT: u8;
}

macro_rules! sitk_impl {
    ($($T:ty: $sitk:expr $(,)?)*) => {
        $(
            impl PixelType for $T {
                const PT: u8 = $sitk;
            }
        )*
    };
}

sitk_impl! {
    u8: 1,
    i8: 2,
    u16: 3,
    i16: 4,
    u32: 5,
    i32: 6,
    u64: 7,
    i64: 8,
    f32: 9,
    f64: 10,
}

#[cfg(target_pointer_width = "64")]
sitk_impl!(usize: 7);
#[cfg(target_pointer_width = "32")]
sitk_impl!(usize: 5);
#[cfg(target_pointer_width = "64")]
sitk_impl!(isize: 8);
#[cfg(target_pointer_width = "32")]
sitk_impl!(isize: 6);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Transform {
    pub parameters: [f64; 6],
    pub dparameters: [f64; 6],
    pub origin: [f64; 2],
    pub shape: [usize; 2],
}

impl Mul for Transform {
    type Output = Transform;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Transform) -> Transform {
        let m = self.matrix().dot(&other.matrix());
        let dm = self.dmatrix().dot(&other.matrix()) + self.matrix().dot(&other.dmatrix());
        Transform {
            parameters: [
                m[[0, 0]],
                m[[0, 1]],
                m[[1, 0]],
                m[[1, 1]],
                m[[2, 0]],
                m[[2, 1]],
            ],
            dparameters: [
                dm[[0, 0]],
                dm[[0, 1]],
                dm[[1, 0]],
                dm[[1, 1]],
                dm[[2, 0]],
                dm[[2, 1]],
            ],
            origin: self.origin,
            shape: self.shape,
        }
    }
}

impl PartialEq<Self> for Transform {
    fn eq(&self, other: &Self) -> bool {
        self.parameters == other.parameters
            && self.dparameters == other.dparameters
            && self.origin == other.origin
            && self.shape == other.shape
    }
}

impl Eq for Transform {}

impl Transform {
    /// parameters: flat 2x2 part of matrix, translation; origin: center of rotation
    pub fn new(parameters: [f64; 6], origin: [f64; 2], shape: [usize; 2]) -> Self {
        Self {
            parameters,
            dparameters: [0f64; 6],
            origin,
            shape,
        }
    }

    /// find the affine transform which transforms moving into fixed
    pub fn register_affine<'a, A, T>(fixed: A, moving: A) -> Result<Transform>
    where
        T: 'a + PixelType,
        A: AsArray<'a, T, Ix2>,
    {
        let (parameters, origin, shape) = register(fixed, moving, true)?;
        Ok(Transform {
            parameters,
            dparameters: [0f64; 6],
            origin,
            shape,
        })
    }

    /// find the translation which transforms moving into fixed
    pub fn register_translation<'a, A, T>(fixed: A, moving: A) -> Result<Transform>
    where
        T: 'a + PixelType,
        A: AsArray<'a, T, Ix2>,
    {
        let (parameters, origin, shape) = register(fixed, moving, false)?;
        Ok(Transform {
            parameters,
            dparameters: [0f64; 6],
            origin,
            shape,
        })
    }

    /// create a transform from a xy translation
    pub fn from_translation(translation: [f64; 2]) -> Self {
        Transform {
            parameters: [1f64, 0f64, 0f64, 1f64, translation[0], translation[1]],
            dparameters: [0f64; 6],
            origin: [0f64; 2],
            shape: [0usize; 2],
        }
    }

    /// read a transform from a file
    pub fn from_file(path: PathBuf) -> Result<Self> {
        let file = File::open(path)?;
        Ok(from_reader(file)?)
    }

    /// write a transform to a file
    pub fn to_file(&self, path: PathBuf) -> Result<()> {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        to_writer(&mut file, self)?;
        Ok(())
    }

    /// true if transform does nothing
    pub fn is_unity(&self) -> bool {
        self.parameters == [1f64, 0f64, 0f64, 1f64, 0f64, 0f64]
    }

    /// transform an image using nearest neighbor interpolation
    pub fn transform_image_bspline<'a, A, T>(&self, image: A) -> Result<Array2<T>>
    where
        T: 'a + PixelType,
        A: AsArray<'a, T, Ix2>,
    {
        interp(self.parameters, self.origin, image, false)
    }

    /// transform an image using bspline interpolation
    pub fn transform_image_nearest_neighbor<'a, A, T>(&self, image: A) -> Result<Array2<T>>
    where
        T: 'a + PixelType,
        A: AsArray<'a, T, Ix2>,
    {
        interp(self.parameters, self.origin, image, true)
    }

    /// get coordinates resulting from transforming input coordinates
    pub fn transform_coordinates<'a, A, T>(&self, coordinates: A) -> Result<Array2<f64>>
    where
        T: 'a + Clone + Into<f64>,
        A: AsArray<'a, T, Ix2>,
    {
        let coordinates = coordinates.into();
        let s = coordinates.shape();
        if s[1] != 2 {
            return Err(anyhow!("coordinates must have two columns"));
        }
        let m = self.matrix();
        let mut res = Array2::zeros([s[0], s[1]]);
        for i in 0..s[0] {
            let a = array![
                coordinates[[i, 0]].clone().into(),
                coordinates[[i, 1]].clone().into(),
                1f64
            ]
            .to_owned();
            let b = m.dot(&a);
            res.slice_mut(s![i, ..]).assign(&b.slice(s![..2]));
        }
        Ok(res)
    }

    /// get the matrix defining the transform
    pub fn matrix(&self) -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 3),
            vec![
                self.parameters[0],
                self.parameters[1],
                self.parameters[4],
                self.parameters[2],
                self.parameters[3],
                self.parameters[5],
                0f64,
                0f64,
                1f64,
            ],
        )
        .unwrap()
    }

    /// get the matrix describing the error of the transform
    pub fn dmatrix(&self) -> Array2<f64> {
        Array2::from_shape_vec(
            (3, 3),
            vec![
                self.dparameters[0],
                self.dparameters[1],
                self.dparameters[4],
                self.dparameters[2],
                self.dparameters[3],
                self.dparameters[5],
                0f64,
                0f64,
                1f64,
            ],
        )
        .unwrap()
    }

    /// get the inverse transform
    pub fn inverse(&self) -> Result<Transform> {
        fn det(a: ArrayView2<f64>) -> f64 {
            (a[[0, 0]] * a[[1, 1]]) - (a[[0, 1]] * a[[1, 0]])
        }

        let m = self.matrix();
        let d = det(m.slice(s![..2, ..2]));
        if d == 0f64 {
            return Err(anyhow!("transform matrix is not invertible"));
        }
        let parameters = [
            det(m.slice(s![1.., 1..])) / d,
            -det(m.slice(s![..;2, 1..])) / d,
            -det(m.slice(s![1.., ..;2])) / d,
            det(m.slice(s![..;2, ..;2])) / d,
            det(m.slice(s![..2, 1..])) / d,
            -det(m.slice(s![..2, ..;2])) / d,
        ];

        Ok(Transform {
            parameters,
            dparameters: [0f64; 6],
            origin: self.origin,
            shape: self.shape,
        })
    }

    /// adapt the transform to a new origin and shape
    pub fn adapt(&mut self, origin: [f64; 2], shape: [usize; 2]) {
        self.origin = [
            origin[0] + (((self.shape[0] - shape[0]) as f64) / 2f64),
            origin[1] + (((self.shape[1] - shape[1]) as f64) / 2f64),
        ];
        self.shape = shape;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use ndarray::Array2;
    use num::Complex;
    use tempfile::NamedTempFile;

    /// An example of generating julia fractals.
    fn julia_image(shift_x: f32, shift_y: f32) -> Result<Array2<u8>> {
        let imgx = 800;
        let imgy = 600;

        let scalex = 3.0 / imgx as f32;
        let scaley = 3.0 / imgy as f32;

        let mut im = Array2::<u8>::zeros((imgy, imgx));
        for x in 0..imgx {
            for y in 0..imgy {
                let cy = (y as f32 + shift_y) * scalex - 1.5;
                let cx = (x as f32 + shift_x) * scaley - 1.5;

                let c = Complex::new(-0.4, 0.6);
                let mut z = Complex::new(cy, cx);

                let mut i = 0;
                while i < 255 && z.norm() <= 2.0 {
                    z = z * z + c;
                    i += 1;
                }

                im[[y, x]] = i as u8;
            }
        }
        Ok(im)
    }

    #[test]
    fn test_serialization() -> Result<()> {
        let file = NamedTempFile::new()?;
        let t = Transform::new([1.2, 0.3, -0.4, 0.9, 10.2, -9.5], [59.5, 49.5], [120, 100]);
        t.to_file(file.path().to_path_buf())?;
        let s = Transform::from_file(file.path().to_path_buf())?;
        assert_eq!(s, t);
        Ok(())
    }

    macro_rules! interp_tests_bspline {
        ($($name:ident: $t:ty $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<()> {
                    let j = julia_image(-120f32, 10f32)?.mapv(|x| x as $t);
                    let k = julia_image(0f32, 0f32)?.mapv(|x| x as $t);
                    let shape = j.shape();
                    let origin = [
                        ((shape[1] - 1) as f64) / 2f64,
                        ((shape[0] - 1) as f64) / 2f64,
                    ];
                    let transform = Transform::new([1., 0., 0., 1., 120., -10.], origin, [shape[0], shape[1]]);
                    let n = transform.transform_image_bspline(j.view())?;
                    let d = (k.mapv(|x| x as f64) - n.mapv(|x| x as f64)).powi(2).sum();
                    assert!(d <= (shape[0] * shape[1]) as f64);
                    Ok(())
                }
            )*
        }
    }

    interp_tests_bspline! {
        interpbs_u8: u8,
        interpbs_i8: i8,
        interpbs_u16: u16,
        interpbs_i16: i16,
        interpbs_u32: u32,
        interpbs_i32: i32,
        interpbs_u64: u64,
        interpbs_i64: i64,
        interpbs_f32: f32,
        interpbs_f64: f64,
    }

    macro_rules! interp_tests_nearest_neighbor {
        ($($name:ident: $t:ty $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<()> {
                    let j = julia_image(-120f32, 10f32)?.mapv(|x| x as $t);
                    let k = julia_image(0f32, 0f32)?.mapv(|x| x as $t);
                    let shape = j.shape();
                    let origin = [
                        ((shape[1] - 1) as f64) / 2f64,
                        ((shape[0] - 1) as f64) / 2f64,
                    ];
                    let j0 = j.clone();
                    let k0 = k.clone();
                    let transform = Transform::new([1., 0., 0., 1., 120., -10.], origin, [shape[0], shape[1]]);
                    // make sure j & k weren't mutated
                    assert!(j.iter().zip(j0.iter()).map(|(a, b)| a == b).all(|x| x));
                    assert!(k.iter().zip(k0.iter()).map(|(a, b)| a == b).all(|x| x));
                    let n = transform.transform_image_nearest_neighbor(j.view())?;
                    let d = (k.mapv(|x| x as f64) - n.mapv(|x| x as f64)).powi(2).sum();
                    assert!(d <= (shape[0] * shape[1]) as f64);
                    Ok(())
                }
            )*
        }
    }

    interp_tests_nearest_neighbor! {
        interpnn_u8: u8,
        interpnn_i8: i8,
        interpnn_u16: u16,
        interpnn_i16: i16,
        interpnn_u32: u32,
        interpnn_i32: i32,
        interpnn_u64: u64,
        interpnn_i64: i64,
        interpnn_f32: f32,
        interpnn_f64: f64,
    }

    macro_rules! registration_tests_translation {
        ($($name:ident: $t:ty $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<()> {
                    let j = julia_image(0f32, 0f32)?.mapv(|x| x as $t);
                    let k = julia_image(10f32, 20f32)?.mapv(|x| x as $t);
                    let j0 = j.clone();
                    let k0 = k.clone();
                    let t = Transform::register_translation(j.view(), k.view())?;
                    // make sure j & k weren't mutated
                    assert!(j.iter().zip(j0.iter()).map(|(a, b)| a == b).all(|x| x));
                    assert!(k.iter().zip(k0.iter()).map(|(a, b)| a == b).all(|x| x));
                    let mut m = Array2::eye(3);
                    m[[0, 2]] = -10f64;
                    m[[1, 2]] = -20f64;
                    let d = (t.matrix() - m).powi(2).sum();
                    assert!(d < 0.01);
                    Ok(())
                }
            )*
        }
    }

    registration_tests_translation! {
        registration_translation_u8: u8,
        registration_translation_i8: i8,
        registration_translation_u16: u16,
        registration_translation_i16: i16,
        registration_translation_u32: u32,
        registration_translation_i32: i32,
        registration_translation_u64: u64,
        registration_translation_i64: i64,
        registration_translation_f32: f32,
        registration_translation_f64: f64,
    }

    macro_rules! registration_tests_affine {
        ($($name:ident: $t:ty $(,)?)*) => {
            $(
                #[test]
                fn $name() -> Result<()> {
                    let j = julia_image(0f32, 0f32)?.mapv(|x| x as $t);
                    let shape = j.shape();
                    let origin = [
                        ((shape[1] - 1) as f64) / 2f64,
                        ((shape[0] - 1) as f64) / 2f64,
                    ];
                    let s = Transform::new([1.2, 0., 0., 1., 5., 7.], origin, [shape[0], shape[1]]);
                    let k = s.transform_image_bspline(j.view())?;
                    let t = Transform::register_affine(j.view(), k.view())?.inverse()?;
                    let d = (t.matrix() - s.matrix()).powi(2).sum();
                    assert!(d < 0.01);
                    Ok(())
                }
            )*
        }
    }

    registration_tests_affine! {
        registration_tests_affine_u8: u8,
        registration_tests_affine_i8: i8,
        registration_tests_affine_u16: u16,
        registration_tests_affine_i16: i16,
        registration_tests_affine_u32: u32,
        registration_tests_affine_i32: i32,
        registration_tests_affine_u64: u64,
        registration_tests_affine_i64: i64,
        registration_tests_affine_f32: f32,
        registration_tests_affine_f64: f64,
    }
}
