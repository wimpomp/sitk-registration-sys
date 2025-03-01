#include <SimpleITK.h>
#include <sitkImageOperators.h>
#include <cstring>

namespace sitk = itk::simple;

using namespace std;


template <typename T> 
sitk::Image make_image(
  unsigned int width,
  unsigned int height,
  T* image,
  sitk::PixelIDValueEnum id
) {
  sitk::Image im(width, height, id);
  if (id == sitk::PixelIDValueEnum::sitkUInt8) {
    uint8_t* b = im.GetBufferAsUInt8();
    memcpy(b, image, width * height);
  } else if (id == sitk::PixelIDValueEnum::sitkInt8) {
    int8_t* b = im.GetBufferAsInt8();
    memcpy(b, image, width * height);
  } else if (id == sitk::PixelIDValueEnum::sitkUInt16) {
    uint16_t* b = im.GetBufferAsUInt16();
    memcpy(b, image, width * height * 2);
  } else if (id == sitk::PixelIDValueEnum::sitkInt16) {
    int16_t* b = im.GetBufferAsInt16();
    memcpy(b, image, width * height * 2);
  } else if (id == sitk::PixelIDValueEnum::sitkUInt32) {
    uint32_t* b = im.GetBufferAsUInt32();
    memcpy(b, image, width * height * 4);
  } else if (id == sitk::PixelIDValueEnum::sitkInt32) {
    int32_t* b = im.GetBufferAsInt32();
    memcpy(b, image, width * height * 4);
  } else if (id == sitk::PixelIDValueEnum::sitkUInt64) {
    uint64_t* b = im.GetBufferAsUInt64();
    memcpy(b, image, width * height * 8);
  } else if (id == sitk::PixelIDValueEnum::sitkInt64) {
    int64_t* b = im.GetBufferAsInt64();
    memcpy(b, image, width * height * 8);
  } else if (id == sitk::PixelIDValueEnum::sitkFloat32) {
    float* b = im.GetBufferAsFloat();
    memcpy(b, image, width * height * 4);
  } else if (id == sitk::PixelIDValueEnum::sitkFloat64) {
    double* b = im.GetBufferAsDouble();
    memcpy(b, image, width * height * 8);
  } 
  return im;
}


sitk::Image
interp(
  double* transform,
  double* origin,
  sitk::Image image,
  bool bspline_or_nn
) {
  try {
    vector<double> matrix = {transform[0], transform[1], transform[2], transform[3]};
    vector<double> translation = {transform[4], transform[5]};
    vector<double> ori = {origin[0], origin[1]};
    sitk::AffineTransform t(matrix, translation, ori);
    sitk::InterpolatorEnum interpolator = (bspline_or_nn == false) ? sitk::sitkBSpline : sitk::sitkNearestNeighbor;
    image = sitk::Resample(image, t, interpolator);
    return image;
  } catch (const std::exception &exc) {
    cerr << exc.what();
    return image;
  }
}


extern "C" void
interp_u8(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  uint8_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkUInt8);
  im = interp(transform, origin, im, bspline_or_nn);
  uint8_t* c = im.GetBufferAsUInt8();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_i8(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  int8_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkInt8);
  im = interp(transform, origin, im, bspline_or_nn);
  int8_t* c = im.GetBufferAsInt8();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_u16(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  uint16_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkUInt16);
  im = interp(transform, origin, im, bspline_or_nn);
  uint16_t* c = im.GetBufferAsUInt16();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_i16(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  int16_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkInt16);
  im = interp(transform, origin, im, bspline_or_nn);
  int16_t* c = im.GetBufferAsInt16();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_u32(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  uint32_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkUInt32);
  im = interp(transform, origin, im, bspline_or_nn);
  uint32_t* c = im.GetBufferAsUInt32();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_i32(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  int32_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkInt32);
  im = interp(transform, origin, im, bspline_or_nn);
  int32_t* c = im.GetBufferAsInt32();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_u64(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  uint64_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkUInt64);
  im = interp(transform, origin, im, bspline_or_nn);
  uint64_t* c = im.GetBufferAsUInt64();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_i64(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  int64_t** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkInt64);
  im = interp(transform, origin, im, bspline_or_nn);
  int64_t* c = im.GetBufferAsInt64();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_f32(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  float** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkFloat32);
  im = interp(transform, origin, im, bspline_or_nn);
  float* c = im.GetBufferAsFloat();
  memcpy(*image, c, width * height);
}

extern "C" void
interp_f64(
  unsigned int width,
  unsigned int height,
  double* transform,
  double* origin,
  double** image,
  bool bspline_or_nn
) {
  sitk::Image im = make_image(width, height, *image, sitk::PixelIDValueEnum::sitkFloat64);
  im = interp(transform, origin, im, bspline_or_nn);
  double* c = im.GetBufferAsDouble();
  memcpy(*image, c, width * height);
}


void
reg(
  sitk::Image fixed,
  sitk::Image moving,
  bool t_or_a,
  double** transform
) {
  try {
    string kind = (t_or_a == false) ? "translation" : "affine";
    sitk::ElastixImageFilter tfilter = sitk::ElastixImageFilter();
    tfilter.LogToConsoleOff();
    tfilter.SetFixedImage(fixed);
    tfilter.SetMovingImage(moving);
    tfilter.SetParameterMap(sitk::GetDefaultParameterMap(kind));
    tfilter.Execute();

    sitk::ElastixImageFilter::ParameterMapType parameter_map = tfilter.GetTransformParameterMap(0);
    for (sitk::ElastixImageFilter::ParameterMapType::iterator parameter = parameter_map.begin(); parameter != parameter_map.end(); ++parameter) {
      if (parameter->first == "TransformParameters") {
        vector<string> tp = parameter->second;
        if (t_or_a == true) {
          for (int j = 0; j < tp.size(); j++) {
            (*transform)[j] = stod(tp[j]);
          }
        } else {
          (*transform)[0] = 1.0;
          (*transform)[1] = 0.0;
          (*transform)[2] = 0.0;
          (*transform)[3] = 1.0;
          for (int j = 0; j < tp.size(); j++) {
            (*transform)[j + 4] = stod(tp[j]);
          }
        }
      }
    } 
  } catch (const std::exception &exc) {
    cerr << exc.what();
  }
} 


extern "C" void
register_u8(
  unsigned int width,
  unsigned int height,
  uint8_t* fixed_arr,
  uint8_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkUInt8;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_i8(
  unsigned int width,
  unsigned int height,
  int8_t* fixed_arr,
  int8_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkInt8;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_u16(
  unsigned int width,
  unsigned int height,
  uint16_t* fixed_arr,
  uint16_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkUInt16;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_i16(
  unsigned int width,
  unsigned int height,
  int16_t* fixed_arr,
  int16_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkInt16;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_u32(
  unsigned int width,
  unsigned int height,
  uint32_t* fixed_arr,
  uint32_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkUInt32;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_i32(
  unsigned int width,
  unsigned int height,
  int32_t* fixed_arr,
  int32_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkInt32;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_u64(
  unsigned int width,
  unsigned int height,
  uint64_t* fixed_arr,
  uint64_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkUInt64;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_i64(
  unsigned int width,
  unsigned int height,
  int64_t* fixed_arr,
  int64_t* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkInt64;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_f32(
  unsigned int width,
  unsigned int height,
  float* fixed_arr,
  float* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkFloat32;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}

extern "C" void
register_f64(
  unsigned int width,
  unsigned int height,
  double* fixed_arr,
  double* moving_arr,
  bool t_or_a,
  double** transform
) {
  sitk::PixelIDValueEnum id = sitk::PixelIDValueEnum::sitkFloat64;
  sitk::Image fixed = make_image(width, height, fixed_arr, id);
  sitk::Image moving = make_image(width, height, moving_arr, id);
  reg(fixed, moving, t_or_a, transform);
}