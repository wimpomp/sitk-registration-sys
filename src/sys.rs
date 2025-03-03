use anyhow::Result;
use libc::{c_double, c_uint};
use ndarray::{Array2, ArrayView2};
use one_at_a_time_please::one_at_a_time;
use std::ptr;

macro_rules! register_fn {
    ($($name:ident: $T:ty $(,)?)*) => {
        $(
            fn $name(
                width: c_uint,
                height: c_uint,
                fixed_arr: *const $T,
                moving_arr: *const $T,
                translation_or_affine: bool,
                transform: &mut *mut c_double,
            );
        )*
    };
}

macro_rules! interp_fn {
    ($($name:ident: $T:ty $(,)?)*) => {
        $(
            fn $name(
                width: c_uint,
                height: c_uint,
                transform: *const c_double,
                origin: *const c_double,
                image: &mut *mut $T,
                bspline_or_nn: bool,
            );
        )*
    };
}

unsafe extern "C" {
    register_fn! {
        register_u8: u8,
        register_i8: i8,
        register_u16: u16,
        register_i16: i16,
        register_u32: u32,
        register_i32: i32,
        register_u64: u64,
        register_i64: i64,
        register_f32: f32,
        register_f64: f64,
    }

    interp_fn! {
        interp_u8: u8,
        interp_i8: i8,
        interp_u16: u16,
        interp_i16: i16,
        interp_u32: u32,
        interp_i32: i32,
        interp_u64: u64,
        interp_i64: i64,
        interp_f32: f32,
        interp_f64: f64,
    }
}

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

#[one_at_a_time]
pub(crate) fn register<T: PixelType>(
    fixed: ArrayView2<T>,
    moving: ArrayView2<T>,
    translation_or_affine: bool,
) -> Result<([f64; 6], [f64; 2], [usize; 2])> {
    let shape: Vec<usize> = fixed.shape().to_vec();
    let width = shape[1] as c_uint;
    let height = shape[0] as c_uint;
    let fixed: Vec<_> = fixed.into_iter().cloned().collect();
    let moving: Vec<_> = moving.into_iter().cloned().collect();
    let fixed_ptr = fixed.as_ptr();
    let moving_ptr = moving.as_ptr();
    let mut transform: Vec<c_double> = vec![0.0; 6];
    let mut transform_ptr: *mut c_double = ptr::from_mut(unsafe { &mut *transform.as_mut_ptr() });

    // let ma0 = &mut moving as *mut Vec<T> as usize;
    // println!("ma0: {:#x}", ma0);

    match T::PT {
        1 => {
            unsafe {
                register_u8(
                    width,
                    height,
                    fixed_ptr as *const u8,
                    moving_ptr as *const u8,
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
                    fixed_ptr as *const i8,
                    moving_ptr as *const i8,
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
                    fixed_ptr as *const u16,
                    moving_ptr as *const u16,
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
                    fixed_ptr as *const i16,
                    moving_ptr as *const i16,
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
                    fixed_ptr as *const u32,
                    moving_ptr as *const u32,
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
                    fixed_ptr as *const i32,
                    moving_ptr as *const i32,
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
                    fixed_ptr as *const u64,
                    moving_ptr as *const u64,
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
                    fixed_ptr as *const i64,
                    moving_ptr as *const i64,
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
                    fixed_ptr as *const f32,
                    moving_ptr as *const f32,
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
                    fixed_ptr as *const f64,
                    moving_ptr as *const f64,
                    translation_or_affine,
                    &mut transform_ptr,
                )
            };
        }
        _ => {}
    }

    // let ma1 = &mut moving as *mut Vec<T> as usize;
    // println!("ma1: {:#x}", ma1);

    // println!("{}", fixed.len());
    // println!("{}", moving.len());

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
