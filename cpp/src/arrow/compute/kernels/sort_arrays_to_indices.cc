// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "arrow/compute/kernels/sort_arrays_to_indices.h"
#include <arrow/util/checked_cast.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <numeric>
#include <vector>

#include "arrow/builder.h"
#include "arrow/compute/context.h"
#include "arrow/compute/expression.h"
#include "arrow/compute/logical_type.h"
#include "arrow/type_traits.h"

namespace arrow {

class Array;

namespace compute {

/// \brief UnaryKernel implementing SortArraysToIndices operation
class ARROW_EXPORT SortArraysToIndicesKernel {
 protected:
  std::shared_ptr<DataType> type_;

 public:
  /// \brief UnaryKernel interface
  ///
  /// delegates to subclasses via SortArraysToIndices()
  virtual Status Call(FunctionContext* ctx, std::vector<std::shared_ptr<Array>> values,
                      std::shared_ptr<Array>* offsets) = 0;

  /// \brief output type of this kernel
  std::shared_ptr<DataType> out_type() const { return uint64(); }

  /// \brief single-array implementation
  virtual Status SortArraysToIndices(FunctionContext* ctx,
                                     std::vector<std::shared_ptr<Array>> values,
                                     std::shared_ptr<Array>* offsets) = 0;

  /// \brief factory for SortArraysToIndicesKernel
  ///
  /// \param[in] value_type constructed SortArraysToIndicesKernel will support sorting
  ///            values of this type
  /// \param[out] out created kernel
  static Status Make(const std::shared_ptr<DataType>& value_type,
                     std::unique_ptr<SortArraysToIndicesKernel>* out, bool nulls_first,
                     bool asc);
};

template <typename ArrowType>
class SortArraysToIndicesKernelImpl : public SortArraysToIndicesKernel {
  using ArrayType = typename TypeTraits<ArrowType>::ArrayType;

 public:
  explicit SortArraysToIndicesKernelImpl(bool nulls_first = true, bool asc = true)
      : nulls_first_(nulls_first), asc_(asc) {}

  Status SortArraysToIndices(FunctionContext* ctx,
                             std::vector<std::shared_ptr<Array>> values,
                             std::shared_ptr<Array>* offsets) {
    return SortArraysToIndicesImpl(ctx, values, offsets);
  }

  Status Call(FunctionContext* ctx, std::vector<std::shared_ptr<Array>> values,
              std::shared_ptr<Array>* offsets) override {
    std::shared_ptr<Array> offsets_array;
    RETURN_NOT_OK(this->SortArraysToIndices(ctx, values, &offsets_array));
    *offsets = offsets_array;
    return Status::OK();
  }

  std::shared_ptr<DataType> out_type() const { return type_; }

 private:
  bool nulls_first_, asc_;
  std::vector<std::shared_ptr<ArrayType>> typed_arrays_;
  uint64_t merge_time = 0;

  std::pair<uint64_t, uint64_t> merge(
      ArrayItemIndex* arrow_buffer, ArrayItemIndex* tmp_buffer,
      std::vector<std::pair<uint64_t, uint64_t>>::iterator arrays_valid_range_begin,
      std::vector<std::pair<uint64_t, uint64_t>>::iterator arrays_valid_range_end) {
    auto size = arrays_valid_range_end - arrays_valid_range_begin;
    std::pair<uint64_t, uint64_t> left;
    std::pair<uint64_t, uint64_t> right;
    if (size > 2) {
      auto half_size = size / 2;
      auto arrays_valid_range_middle = arrays_valid_range_begin + half_size;
      left = merge(arrow_buffer, tmp_buffer, arrays_valid_range_begin,
                   arrays_valid_range_middle);
      right = merge(arrow_buffer, tmp_buffer, arrays_valid_range_middle,
                    arrays_valid_range_end);
    } else if (size == 2) {
      left = *arrays_valid_range_begin;
      right = *(arrays_valid_range_end - 1);
    } else {
      // only one item
      return *arrays_valid_range_begin;
    }
    auto left_size = left.second - left.first;
    auto right_size = right.second - right.first;

    // printf("left.first is %ld, left.size is %ld, right.first is %ld, right.size is
    // %ld\n",
    //       left.first, left_size, right.first, right_size);
    memcpy(tmp_buffer + left.first, arrow_buffer + left.first,
           left_size * sizeof(ArrayItemIndex));
    memcpy(tmp_buffer + right.first, arrow_buffer + right.first,
           right_size * sizeof(ArrayItemIndex));

    if (asc_) {
      std::set_union(tmp_buffer + left.first, tmp_buffer + left.second,
                     tmp_buffer + right.first, tmp_buffer + right.second,
                     arrow_buffer + left.first,
                     [this](ArrayItemIndex left, ArrayItemIndex right) {
                       return typed_arrays_[left.array_id]->GetView(left.id) <
                              typed_arrays_[right.array_id]->GetView(right.id);
                     });
    } else {
      std::set_union(tmp_buffer + left.first, tmp_buffer + left.second,
                     tmp_buffer + right.first, tmp_buffer + right.second,
                     arrow_buffer + left.first,
                     [this](ArrayItemIndex left, ArrayItemIndex right) {
                       return typed_arrays_[left.array_id]->GetView(left.id) >
                              typed_arrays_[right.array_id]->GetView(right.id);
                     });
    }

    return std::make_pair(left.first, right.second);
  }

  Status SortArraysToIndicesImpl(FunctionContext* ctx,
                                 std::vector<std::shared_ptr<Array>> values,
                                 std::shared_ptr<Array>* offsets) {
    // initiate buffer for all arrays
    std::shared_ptr<Buffer> indices_buf;
    int64_t items_total = 0;
    int64_t nulls_total = 0;
    for (auto array : values) {
      items_total += array->length();
      nulls_total += array->null_count();
    }
    int64_t buf_size = items_total * sizeof(ArrayItemIndex);
    RETURN_NOT_OK(AllocateBuffer(ctx->memory_pool(), sizeof(ArrayItemIndex) * buf_size,
                                 &indices_buf));

    // start to partition not_null with null
    ArrayItemIndex* indices_begin =
        reinterpret_cast<ArrayItemIndex*>(indices_buf->mutable_data());
    ArrayItemIndex* indices_end = indices_begin + items_total;
    std::vector<std::pair<uint64_t, uint64_t>> arrays_valid_range;

    int64_t array_id = 0;
    int64_t indices_i = 0;
    int64_t indices_null = 0;

    // we should support nulls first and nulls last here
    // we should also support desc and asc here

    for (auto array : values) {
      auto typed_array = std::dynamic_pointer_cast<ArrayType>(array);
      typed_arrays_.push_back(typed_array);
      int64_t array_begin = 0;
      int64_t array_end = 0;
      if (nulls_first_) {
        array_begin = nulls_total + indices_i;
        for (int64_t i = 0; i < array->length(); i++) {
          if (!array->IsNull(i)) {
            (indices_begin + nulls_total + indices_i)->array_id = array_id;
            (indices_begin + nulls_total + indices_i)->id = i;
            indices_i++;
          } else {
            (indices_begin + indices_null)->array_id = array_id;
            (indices_begin + indices_null)->id = i;
            indices_null++;
          }
        }
        array_end = nulls_total + indices_i;
        arrays_valid_range.push_back(
            std::make_pair(array_begin - nulls_total, array_end - nulls_total));
      } else {
        array_begin = indices_i;
        for (int64_t i = 0; i < array->length(); i++) {
          if (!array->IsNull(i)) {
            (indices_begin + indices_i)->array_id = array_id;
            (indices_begin + indices_i)->id = i;
            indices_i++;
          } else {
            (indices_end - nulls_total + indices_null)->array_id = array_id;
            (indices_end - nulls_total + indices_null)->id = i;
            indices_null++;
          }
        }
        array_end = indices_i;
        arrays_valid_range.push_back(std::make_pair(array_begin, array_end));
      }
      // first round sort
      if (asc_) {
        std::stable_sort(indices_begin + array_begin, indices_begin + array_end,
                         [typed_array, this](ArrayItemIndex left, ArrayItemIndex right) {
                           return typed_array->GetView(left.id) <
                                  typed_array->GetView(right.id);
                         });
      } else {
        std::stable_sort(indices_begin + array_begin, indices_begin + array_end,
                         [typed_array, this](ArrayItemIndex left, ArrayItemIndex right) {
                           return typed_array->GetView(left.id) >
                                  typed_array->GetView(right.id);
                         });
      }
      array_id++;
    }

    // merge sort
    ArrayItemIndex* tmp_buffer_begin = new ArrayItemIndex[indices_i]();
    if (nulls_first_) {
      merge(indices_begin + nulls_total, tmp_buffer_begin, arrays_valid_range.begin(),
            arrays_valid_range.end());
    } else {
      merge(indices_begin, tmp_buffer_begin, arrays_valid_range.begin(),
            arrays_valid_range.end());
    }
    delete[] tmp_buffer_begin;

    auto out_type =
        std::make_shared<FixedSizeBinaryType>(sizeof(ArrayItemIndex) / sizeof(int32_t));
    *offsets = std::make_shared<FixedSizeBinaryArray>(out_type, items_total, indices_buf);
    return Status::OK();
  }
};

template <typename ArrowType>
SortArraysToIndicesKernelImpl<ArrowType>* MakeSortArraysToIndicesKernelImpl(
    bool nulls_first = true, bool asc = true) {
  return new SortArraysToIndicesKernelImpl<ArrowType>(nulls_first, asc);
}

#define PROCESS_SUPPORTED_TYPES(PROCESS) \
  PROCESS(arrow::UInt8Type)              \
  PROCESS(arrow::Int8Type)               \
  PROCESS(arrow::UInt16Type)             \
  PROCESS(arrow::Int16Type)              \
  PROCESS(arrow::UInt32Type)             \
  PROCESS(arrow::Int32Type)              \
  PROCESS(arrow::UInt64Type)             \
  PROCESS(arrow::Int64Type)              \
  PROCESS(arrow::FloatType)              \
  PROCESS(arrow::DoubleType)             \
  PROCESS(arrow::BinaryType)             \
  PROCESS(arrow::StringType)
Status SortArraysToIndicesKernel::Make(const std::shared_ptr<DataType>& value_type,
                                       std::unique_ptr<SortArraysToIndicesKernel>* out,
                                       bool nulls_first, bool asc) {
  SortArraysToIndicesKernel* kernel;
  switch (value_type->id()) {
#define PROCESS(InType)                                                   \
  case InType::type_id: {                                                 \
    kernel = MakeSortArraysToIndicesKernelImpl<InType>(nulls_first, asc); \
  } break;
    PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
    default:
      return Status::NotImplemented("Sorting of ", *value_type, " arrays");
  }
  out->reset(kernel);
  return Status::OK();
}
#undef PROCESS_SUPPORTED_TYPES

Status SortArraysToIndices(FunctionContext* ctx,
                           std::vector<std::shared_ptr<Array>> values,
                           std::shared_ptr<Array>* offsets, bool nulls_first = true,
                           bool asc = true) {
  if (values.size() == 0) {
    return Status::Invalid("Input ArrayList is empty");
  }
  std::unique_ptr<SortArraysToIndicesKernel> kernel;
  RETURN_NOT_OK(
      SortArraysToIndicesKernel::Make(values[0]->type(), &kernel, nulls_first, asc));
  return kernel->Call(ctx, values, offsets);
}

}  // namespace compute
}  // namespace arrow
