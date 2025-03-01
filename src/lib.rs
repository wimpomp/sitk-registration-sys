mod sys;

use crate::sys::{PixelType, interp, register};
use anyhow::{Result, anyhow};
use ndarray::{Array2, ArrayView2, array, s};
use std::ops::Mul;
use std::path::PathBuf;

#[derive(Clone, Debug)]
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
    pub fn register_affine<T: PixelType>(
        fixed: ArrayView2<T>,
        moving: ArrayView2<T>,
    ) -> Result<Transform> {
        let (parameters, origin, shape) = register(fixed, moving, true)?;
        Ok(Transform {
            parameters,
            dparameters: [0f64; 6],
            origin,
            shape,
        })
    }

    /// find the translation which transforms moving into fixed
    pub fn register_translation<T: PixelType>(
        fixed: ArrayView2<T>,
        moving: ArrayView2<T>,
    ) -> Result<Transform> {
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
    pub fn from_file(file: PathBuf) -> Result<Self> {
        todo!()
    }

    /// write a transform to a file
    pub fn to_file(&self, file: PathBuf) -> Result<()> {
        todo!()
    }

    /// true if transform does nothing
    pub fn is_unity(&self) -> bool {
        self.parameters == [1f64, 0f64, 0f64, 1f64, 0f64, 0f64]
    }

    /// transform an image using nearest neighbor interpolation
    pub fn transform_image_bspline<T: PixelType>(&self, image: ArrayView2<T>) -> Result<Array2<T>> {
        interp(self.parameters, self.origin, image, false)
    }

    /// transform an image using bspline interpolation
    pub fn transform_image_nearest_neighbor<T: PixelType>(
        &self,
        image: ArrayView2<T>,
    ) -> Result<Array2<T>> {
        interp(self.parameters, self.origin, image, true)
    }

    /// get coordinates resulting from transforming input coordinates
    pub fn transform_coordinates<T>(&self, coordinates: ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Into<f64>,
    {
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
        let d0 = det(m.slice(s![1.., 1..]));
        if d0 == 0f64 {
            return Err(anyhow!("transform matrix is not invertible"));
        }
        let d2 = det(m.slice(s![..2, ..2]));
        let parameters = [
            d0 / d2,
            -det(m.slice(s![..;2, 1..])) / d2,
            -det(m.slice(s![1.., ..;2])) / d2,
            det(m.slice(s![..;2, ..;2])) / d2,
            det(m.slice(s![..2, 1..])) / d2,
            -det(m.slice(s![..2, ..;2])) / d2,
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
    use tiffwrite::IJTiffFile;

    /// An example of generating julia fractals.
    fn julia_image() -> Result<Array2<u8>> {
        let imgx = 800;
        let imgy = 600;

        let scalex = 3.0 / imgx as f32;
        let scaley = 3.0 / imgy as f32;

        let mut im = Array2::<u8>::zeros((imgy, imgx));
        for x in 0..imgx {
            for y in 0..imgy {
                let cx = y as f32 * scalex - 1.5;
                let cy = x as f32 * scaley - 1.5;

                let c = Complex::new(-0.4, 0.6);
                let mut z = Complex::new(cx, cy);

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
    fn test_interp() -> Result<()> {
        let j = julia_image()?;
        let mut tif = IJTiffFile::new("interp_test.tif")?;
        tif.save(&j, 0, 0, 0)?;
        let shape = j.shape();
        let origin = [
            ((shape[1] - 1) as f64) / 2f64,
            ((shape[0] - 1) as f64) / 2f64,
        ];
        let transform = Transform::new([1.2, 0., 0., 1., 10., 0.], origin, [shape[0], shape[1]]);
        let k = transform.transform_image_bspline(j.view())?;
        tif.save(&k, 1, 0, 0)?;

        let t = Transform::register_affine(k.view(), j.view())?;
        println!("t: {:#?}", t);
        println!("m: {:#?}", t.matrix());
        println!("i: {:#?}", t.inverse()?.matrix());
        Ok(())
    }
}
