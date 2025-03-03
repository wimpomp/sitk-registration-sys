# sitk-registration-sys

This crate does two things: 
- find an affine transform or translation that transforms one image into the other
- use bpline or nearest neighbor interpolation to apply a transformation to an image

To do this, [SimpleITK](https://github.com/SimpleITK/SimpleITK.git), which is written in
C++, is used. An adapter library is created to expose the required functionality in SimpleITK
in a shared library. Because of this, compilation of this crate requires quite some time, as
wel as cmake.

## Examples
### Registration
```
    let image_a = (some Array2);
    let iameg_b = (some transformed Array2);
    let transform = Transform::register_affine(image_a.view(), image_b.view())?;
    println!("transform: {:#?}", transform);
```

### Interpolation
```    
    let image = (Some Array2);
    let shape = image.shape();
    let origin = [
        ((shape[1] - 1) as f64) / 2f64,
        ((shape[0] - 1) as f64) / 2f64,
    ];
    let transform = Transform::new([1.2, 0., 0., 1., 10., 0.], origin, [shape[0], shape[1]]);
    let transformed_image = transform.transform_image_bspline(image.view())?;
```