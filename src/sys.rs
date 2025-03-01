use anyhow::Result;
use libc::{c_double, c_uint};
use ndarray::{Array2, ArrayView2};
use std::ptr;

macro_rules! register_fn {
    ($T:ty, $t:ident) => {
        fn $t(
            width: c_uint,
            height: c_uint,
            fixed_arr: *const $T,
            moving_arr: *const $T,
            translation_or_affine: bool,
            transform: &mut *mut c_double,
        );
    };
}

macro_rules! interp_fn {
    ($T:ty, $t:ident) => {
        fn $t(
            width: c_uint,
            height: c_uint,
            transform: *const c_double,
            origin: *const c_double,
            image: &mut *mut $T,
            bspline_or_nn: bool,
        );
    };
}

unsafe extern "C" {
    register_fn!(u8, register_u8);
    register_fn!(i8, register_i8);
    register_fn!(u16, register_u16);
    register_fn!(i16, register_i16);
    register_fn!(u32, register_u32);
    register_fn!(i32, register_i32);
    register_fn!(u64, register_u64);
    register_fn!(i64, register_i64);
    register_fn!(f32, register_f32);
    register_fn!(f64, register_f64);
    interp_fn!(u8, interp_u8);
    interp_fn!(i8, interp_i8);
    interp_fn!(u16, interp_u16);
    interp_fn!(i16, interp_i16);
    interp_fn!(u32, interp_u32);
    interp_fn!(i32, interp_i32);
    interp_fn!(u64, interp_u64);
    interp_fn!(i64, interp_i64);
    interp_fn!(f32, interp_f32);
    interp_fn!(f64, interp_f64);
}

pub trait PixelType: Clone {
    const PT: u8;
}

macro_rules! sitk_impl {
    ($T:ty, $sitk:expr) => {
        impl PixelType for $T {
            const PT: u8 = $sitk;
        }
    };
}

sitk_impl!(u8, 1);
sitk_impl!(i8, 2);
sitk_impl!(u16, 3);
sitk_impl!(i16, 4);
sitk_impl!(u32, 5);
sitk_impl!(i32, 6);
sitk_impl!(u64, 7);
sitk_impl!(i64, 8);
#[cfg(target_pointer_width = "64")]
sitk_impl!(usize, 7);
#[cfg(target_pointer_width = "32")]
sitk_impl!(usize, 5);
#[cfg(target_pointer_width = "64")]
sitk_impl!(isize, 8);
#[cfg(target_pointer_width = "32")]
sitk_impl!(isize, 6);
sitk_impl!(f32, 9);
sitk_impl!(f64, 10);

pub(crate) fn interp<T: PixelType>(
    parameters: [f64; 6],
    origin: [f64; 2],
    image: ArrayView2<T>,
    bspline_or_nn: bool,
) -> Result<Array2<T>> {
    let shape: Vec<usize> = image.shape().to_vec();
    let width = shape[1] as c_uint;
    let height = shape[0] as c_uint;
    let mut im: Vec<_> = image.into_iter().cloned().collect();
    let im_ptr: *mut T = ptr::from_mut(unsafe { &mut *im.as_mut_ptr() });

    match T::PT {
        1 => unsafe {
            interp_u8(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut u8),
                bspline_or_nn,
            );
        },
        2 => unsafe {
            interp_i8(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut i8),
                bspline_or_nn,
            );
        },
        3 => unsafe {
            interp_u16(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut u16),
                bspline_or_nn,
            );
        },
        4 => unsafe {
            interp_i16(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut i16),
                bspline_or_nn,
            );
        },
        5 => unsafe {
            interp_u32(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut u32),
                bspline_or_nn,
            );
        },
        6 => unsafe {
            interp_i32(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut i32),
                bspline_or_nn,
            );
        },
        7 => unsafe {
            interp_u64(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut u64),
                bspline_or_nn,
            );
        },
        8 => unsafe {
            interp_i64(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut i64),
                bspline_or_nn,
            );
        },
        9 => unsafe {
            interp_f32(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut f32),
                bspline_or_nn,
            );
        },
        10 => unsafe {
            interp_f64(
                width,
                height,
                parameters.as_ptr(),
                origin.as_ptr(),
                &mut (im_ptr as *mut f64),
                bspline_or_nn,
            );
        },
        _ => {}
    }
    Ok(Array2::from_shape_vec(
        (shape[0], shape[1]),
        im.into_iter().collect(),
    )?)
}

pub(crate) fn register<T: PixelType>(
    fixed: ArrayView2<T>,
    moving: ArrayView2<T>,
    translation_or_affine: bool,
) -> Result<([f64; 6], [f64; 2], [usize; 2])> {
    let shape: Vec<usize> = fixed.shape().to_vec();
    let width = shape[1] as c_uint;
    let height = shape[0] as c_uint;
    let fixed: Vec<_> = fixed.into_iter().collect();
    let moving: Vec<_> = moving.into_iter().collect();
    let mut transform: Vec<c_double> = vec![0.0; 6];
    let mut transform_ptr: *mut c_double = ptr::from_mut(unsafe { &mut *transform.as_mut_ptr() });

    match T::PT {
        1 => {
            unsafe {
                register_u8(
                    width,
                    height,
                    fixed.as_ptr() as *const u8,
                    moving.as_ptr() as *const u8,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        2 => {
            unsafe {
                register_i8(
                    width,
                    height,
                    fixed.as_ptr() as *const i8,
                    moving.as_ptr() as *const i8,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        3 => {
            unsafe {
                register_u16(
                    width,
                    height,
                    fixed.as_ptr() as *const u16,
                    moving.as_ptr() as *const u16,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        4 => {
            unsafe {
                register_i16(
                    width,
                    height,
                    fixed.as_ptr() as *const i16,
                    moving.as_ptr() as *const i16,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        5 => {
            unsafe {
                register_u32(
                    width,
                    height,
                    fixed.as_ptr() as *const u32,
                    moving.as_ptr() as *const u32,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        6 => {
            unsafe {
                register_i32(
                    width,
                    height,
                    fixed.as_ptr() as *const i32,
                    moving.as_ptr() as *const i32,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        7 => {
            unsafe {
                register_u64(
                    width,
                    height,
                    fixed.as_ptr() as *const u64,
                    moving.as_ptr() as *const u64,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        8 => {
            unsafe {
                register_i64(
                    width,
                    height,
                    fixed.as_ptr() as *const i64,
                    moving.as_ptr() as *const i64,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        9 => {
            unsafe {
                register_f32(
                    width,
                    height,
                    fixed.as_ptr() as *const f32,
                    moving.as_ptr() as *const f32,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        10 => {
            unsafe {
                register_f64(
                    width,
                    height,
                    fixed.as_ptr() as *const f64,
                    moving.as_ptr() as *const f64,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        _ => {}
    }

    Ok((
        [
            transform[0] as f64,
            transform[1] as f64,
            transform[2] as f64,
            transform[3] as f64,
            transform[4] as f64,
            transform[5] as f64,
        ],
        [
            ((shape[0] - 1) as f64) / 2f64,
            ((shape[1] - 1) as f64) / 2f64,
        ],
        [shape[0], shape[1]],
    ))
}
