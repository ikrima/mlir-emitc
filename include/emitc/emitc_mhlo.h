// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This file defines functions used by EmitC

#ifndef EMITC_MHLO_H
#define EMITC_MHLO_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <random>
#include <type_traits>
#include <vector>

#include "emitc_tensor.h"

namespace mhlo {
/// See
/// https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/hlo_evaluator.cc
/// for the XLA implementation

/// Functions for MHLO unary elementwise ops
// AbsOp
// TODO support complex numbers
template <typename Src>
inline Src abs(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::abs);

  return unary<Src>(x, f);
}

// CeilOp
template <typename Src>
inline Src ceil(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::ceil);

  return unary<Src>(x, f);
}

// BitcastConvertOp
template <typename Dest, typename Src>
inline Dest bitcast_convert(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  static_assert(sizeof(ET_Src) == sizeof(ET_Dest),
                "Can only bitcast on types of the same size");

  auto cast = [](ET_Src value) {
    ET_Dest result;
    memcpy(&result, &value, sizeof(ET_Src));
    return result;
  };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
}

// CompareOp
template <typename Src, template <typename> class Compare>
typename replace_element_type<bool, Src>::type compare(Src x, Src y) {
  using Dest = typename replace_element_type<bool, Src>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cmp = Compare<ET_Src>{};

  return binary<Dest, Src>(x, y, cmp);
}

// ConvertOp
template <typename Dest, typename Src>
inline Dest convert(Src x) {
  using ET_Dest = typename get_element_type<Dest>::type;
  using ET_Src = typename get_element_type<Src>::type;

  auto cast = [](ET_Src value) { return static_cast<ET_Dest>(value); };

  return unary<Dest, Src, UnaryFuncType<ET_Dest, ET_Src>>(x, cast);
}

// CosOp
template <typename Src>
inline Src cos(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::cos);

  return unary<Src>(x, f);
}

// ExpOp
template <typename Src>
inline Src exponential(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::exp);

  return unary<Src>(x, f);
}

// FloorOp
template <typename Src>
inline Src floor(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::floor);

  return unary<Src>(x, f);
}

// IsFiniteOp
template <typename Src>
inline typename replace_element_type<bool, Src>::type is_finite(Src x) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_floating_point<ET_Src>::value,
                "Operation supports only floating point types");

  using Dest = typename replace_element_type<bool, Src>::type;

  auto f = static_cast<bool (*)(ET_Src)>(std::isfinite);

  return unary<Dest, Src>(x, f);
}

// LogOp
template <typename Src>
inline Src log(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::log);

  return unary<Src>(x, f);
}

// NegOp
template <typename Src>
inline Src negate(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::negate<ET_Src>{};

  return unary<Src>(x, f);
}

// SinOp
template <typename Src>
inline Src sin(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::sin);

  return unary<Src>(x, f);
}

// SqrtOp
template <typename Src>
inline Src sqrt(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::sqrt);

  return unary<Src>(x, f);
}

// TanhOp
template <typename Src>
inline Src tanh(Src x) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = static_cast<ET_Src (*)(ET_Src)>(std::tanh);

  return unary<Src>(x, f);
}

/// Functions for MHLO binary elementwise ops.
// AddOp
template <typename Src>
inline Src add(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::plus<ET_Src>{};

  return binary<Src>(x, y, f);
}

// DivOp
template <typename Src>
inline Src div(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::divides<ET_Src>{};

  return binary<Src>(x, y, f);
}

// MaxOp
template <typename Src>
inline Src max(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::max);

  return binary<Src>(x, y, f);
}

// MinOp
template <typename Src>
inline Src min(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f =
      static_cast<const ET_Src &(*)(const ET_Src &, const ET_Src &)>(std::min);

  return binary<Src>(x, y, f);
}

// MulOp
template <typename Src>
inline Src mul(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::multiplies<ET_Src>{};

  return binary<Src>(x, y, f);
}

// PowOp
template <typename Src>
inline Src pow(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) -> ET_Src {
    if (std::is_integral<ET_Src>::value) {
      const bool negative = b < 0;
      if (b < 0) {
        b = -b;
      }

      ET_Src result = 1;

      for (ET_Src i = 0; i < b; i++) {
        result *= a;
      }

      if (negative) {
        result = 1 / result;
      }
      return result;
    } else {
      return std::pow(a, b);
    }
  };

  return binary<Src>(x, y, f);
}

// ShiftLeftOp
template <typename Src>
inline Src shift_left(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_unsigned<ET_Src>::value,
                "Operation not implemented for signed types");

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a << b; };

  return binary<Src>(x, y, f);
}

// ShiftRightLogicalOp
template <typename Src>
inline Src shift_right_logical(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;
  static_assert(std::is_unsigned<ET_Src>::value,
                "Operation not implemented for signed types");

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a >> b; };

  return binary<Src>(x, y, f);
}

// SubOp
template <typename Src>
inline Src sub(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::minus<ET_Src>{};

  return binary<Src>(x, y, f);
}

/// Functions for MHLO binary logical elementwise ops.
// OrOp
template <typename Src>
inline Src logical_or(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = std::logical_or<ET_Src>{};

  return binary<Src>(x, y, f);
}

// XorOp
template <typename Src>
inline Src logical_xor(Src x, Src y) {
  using ET_Src = typename get_element_type<Src>::type;

  auto f = [](ET_Src a, ET_Src b) -> ET_Src { return a != b; };

  return binary<Src>(x, y, f);
}
/// Functions for other MHLO ops.
// BroadcastInDimOp
template <typename T>
inline std::vector<T> broadcast_in_dim(std::vector<T> x, size_t n) {
  std::vector<T> z;

  for (size_t i = 0; i < n; i++) {
    z.insert(z.end(), x.begin(), x.end());
  }

  return z;
}

// ConcatenateOp
template <typename T>
inline std::vector<T> concatenate(std::vector<T> x, std::vector<T> y) {
  std::vector<T> z(x);
  z.insert(z.end(), y.begin(), y.end());
  return z;
}

// SliceOp
// Overload for 1d case
template <typename T, size_t Start, size_t Limit, size_t Stride,
          size_t InputShape, size_t OutputShape>
std::vector<T> slice(std::vector<T> x) {
  std::vector<T> result(OutputShape);

  size_t idx = 0;
  for (size_t i = Start; i < Limit; i += Stride) {
    result[idx++] = x[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t Start1, size_t Start2, size_t Limit1,
          size_t Limit2, size_t Stride1, size_t Stride2, size_t InputShape1,
          size_t InputShape2, size_t OutputShape1, size_t OutputShape2>
std::vector<T> slice(std::vector<T> x) {
  std::vector<T> result(OutputShape1 * OutputShape2);

  size_t idx = 0;
  for (size_t i = Start1; i < Limit1; i += Stride1) {
    for (size_t j = Start2; j < Limit2; j += Stride2) {
      result[idx++] = x[i * InputShape2 + j];
    }
  }
  return result;
}

// DynamicSliceOp
// Overload for 1d case
template <typename T, size_t Size, size_t InputShape, size_t OutputShape>
std::vector<T> dynamic_slice(std::vector<T> x,
                             std::vector<int64_t> startIndex) {
  std::vector<T> result(OutputShape);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndex_ = startIndex[0];
  startIndex_ = clamp(startIndex_, 0, InputShape - Size);

  size_t limit = startIndex_ + Size;

  size_t idx = 0;
  for (size_t i = startIndex_; i < limit; i++) {
    result[idx++] = x[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t SizeX, size_t SizeY, size_t InputShapeX,
          size_t InputShapeY, size_t OutputShapeX, size_t OutputShapeY>
std::vector<T> dynamic_slice(std::vector<T> x, std::vector<int64_t> startIndexX,
                             std::vector<int64_t> startIndexY) {
  std::vector<T> result(OutputShapeX * OutputShapeY);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndexX_ = startIndexX[0];
  size_t startIndexY_ = startIndexY[0];
  startIndexX_ = clamp(startIndexX_, 0, InputShapeX - SizeX);
  startIndexY_ = clamp(startIndexY_, 0, InputShapeY - SizeY);

  size_t limitX = startIndexX_ + SizeX;
  size_t limitY = startIndexY_ + SizeY;

  size_t idx = 0;
  for (size_t i = startIndexX_; i < limitX; i++) {
    for (size_t j = startIndexY_; j < limitY; j++) {
      result[idx++] = x[i * InputShapeY + j];
    }
  }
  return result;
}

// DynamicUpdateSliceOp
// Overload for 1d case
template <typename T, size_t InputShape, size_t UpdateShape>
std::vector<T> dynamic_update_slice(std::vector<T> x, std::vector<T> u,
                                    std::vector<int64_t> startIndex) {
  std::vector<T> result(x);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndex_ = startIndex[0];
  startIndex_ = clamp(startIndex_, 0, InputShape - UpdateShape);

  for (size_t i = 0; i < UpdateShape; i++) {
    result[startIndex_ + i] = u[i];
  }
  return result;
}

// Overload for 2d case
template <typename T, size_t InputShapeX, size_t InputShapeY,
          size_t UpdateShapeX, size_t UpdateShapeY>
std::vector<T> dynamic_update_slice(std::vector<T> x, std::vector<T> u,
                                    std::vector<int64_t> startIndexX,
                                    std::vector<int64_t> startIndexY) {
  std::vector<T> result(x);

  auto clamp = [](size_t value, size_t minValue, size_t maxValue) {
    return std::max(minValue, std::min(maxValue, value));
  };

  size_t startIndexX_ = startIndexX[0];
  size_t startIndexY_ = startIndexY[0];
  startIndexX_ = clamp(startIndexX_, 0, InputShapeX - UpdateShapeX);
  startIndexY_ = clamp(startIndexY_, 0, InputShapeY - UpdateShapeY);

  for (size_t i = 0; i < UpdateShapeX; i++) {
    for (size_t j = 0; j < UpdateShapeY; j++) {
      result[(startIndexX_ + i) * InputShapeY + (startIndexY_ + j)] =
          u[i * UpdateShapeY + j];
    }
  }
  return result;
}

// ReshapeOp
template <typename Dest, typename Src>
inline Dest reshape(Src x) {
  static_assert(is_tensor<Src>::value, "Expected tensor argument");
  static_assert(is_tensor<Dest>::value, "Expected tensor result");

  using ET_Src = typename get_element_type<Src>::type;
  using ET_Dest = typename get_element_type<Dest>::type;

  static_assert(std::is_same<ET_Src, ET_Dest>::value, "Element type mismatch");
  static_assert(Src::size_ == Dest::size_, "Tensor size mismatch");

  Dest z;

  std::copy(x.begin(), x.end(), z.begin());

  return z;
}

// SelectOp
template <typename Src, IsScalar<Src> = true>
inline Src select(typename replace_element_type<bool, Src>::type pred,
                  Src on_true, Src on_false) {
  static_assert(is_scalar<Src>::value);

  return pred ? on_true : on_false;
}

template <typename Src, IsTensor<Src> = true>
inline Src select(typename replace_element_type<bool, Src>::type pred,
                  Src on_true, Src on_false) {
  static_assert(is_tensor<Src>::value);

  Src z;

  for (size_t i = 0; i < Src::size_; i++) {
    z[i] = pred[i] ? on_true[i] : on_false[i];
  }

  return z;
}

// RngUniformOp
template <typename T>
using IsIntegral =
    typename std::enable_if<std::is_integral<T>::value, bool>::type;
template <typename T>
using IsFloatingPoint =
    typename std::enable_if<std::is_floating_point<T>::value, bool>::type;

// integer types
template <typename T, IsIntegral<T> = true>
std::vector<T> rng_uniform(T low, T high, std::vector<int64_t> shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());

  std::random_device rd;
  std::mt19937 gen(rd());
  // high value is exclusive in xla but inclusive in cpp
  // see https://www.tensorflow.org/xla/operation_semantics?hl=en#rnguniform and
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution
  std::uniform_int_distribution<T> distribution(low, high - 1);
  std::vector<T> result(n);
  for (int64_t i = 0; i < n; i++) {
    result[i] = distribution(gen);
  }
  return result;
}

// floating point types
template <typename T, IsFloatingPoint<T> = true>
std::vector<T> rng_uniform(T low, T high, std::vector<int64_t> shape) {
  int64_t n = std::accumulate(shape.begin(), shape.end(), 1,
                              std::multiplies<int64_t>());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> distribution(low, high);
  std::vector<T> result(n);
  for (int64_t i = 0; i < n; i++) {
    result[i] = distribution(gen);
  }
  return result;
}

// RngBitGeneratorOp
template <typename T, int32_t Algorithm, int64_t N>
std::tuple<std::vector<uint64_t>, std::vector<T>>
rng_bit_generator(std::vector<uint64_t> state) {
  // TODO implement correct algorithm; starting point would be
  // https://github.com/tensorflow/tensorflow/blob/6f59650012f8904745dffaba540afc794c6613be/tensorflow/compiler/xla/service/rng_bit_generator_expander.cc#L56
  std::vector<uint64_t> newState(state);
  std::vector<int64_t> shape{N};

  T min = std::numeric_limits<T>::min();
  T max = std::numeric_limits<T>::max();
  std::vector<T> resultVector = rng_uniform<T>(min, max, shape);

  return std::make_tuple(newState, resultVector);
}

} // namespace mhlo

#endif // EMITC_MHLO_H