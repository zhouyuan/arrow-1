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

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "arrow/compute/context.h"
#include "arrow/compute/kernels/sort_arrays_to_indices.h"
#include "arrow/compute/test_util.h"
#include "arrow/pretty_print.h"
#include "arrow/testing/gtest_common.h"
#include "arrow/testing/gtest_util.h"
#include "arrow/testing/util.h"
#include "arrow/type_traits.h"

namespace arrow {
namespace compute {

template <typename ArrowType>
class TestSortToIndicesKernel : public ComputeFixture, public TestBase {
 private:
  void AssertSortArraysToIndices(std::vector<std::shared_ptr<Array>> values,
                                 const std::shared_ptr<Array> expected) {
    std::shared_ptr<Array> actual;
    ASSERT_OK(arrow::compute::SortArraysToIndices(&this->ctx_, values, &actual));
    ASSERT_OK(actual->ValidateFull());
    AssertArraysEqual(*expected, *actual);
  }

  void SortArraysToIndices(std::vector<std::shared_ptr<Array>> values) {
    std::shared_ptr<Array> actual;
    std::cout << "=============  Input is ===============" << std::endl;
    for (auto array : values) {
      PrettyPrint(*array.get(), 2, &std::cout);
    }
    ASSERT_OK(arrow::compute::SortArraysToIndices(&this->ctx_, values, &actual));
    std::cout << std::endl << "=============  Output is ===============" << std::endl;
    // print res
    ArrayItemIndex* actual_values =
        (ArrayItemIndex*)std::dynamic_pointer_cast<FixedSizeBinaryArray>(actual)
            ->raw_values();
    for (int i = 0; i < actual->length(); i++) {
      ArrayItemIndex* item = actual_values + i;
      if (!values[item->array_id]->IsNull(item->id)) {
        using ArrayType = typename TypeTraits<ArrowType>::ArrayType;
        auto typed_array = std::dynamic_pointer_cast<ArrayType>(values[item->array_id]);
        std::cout << typed_array->GetView(item->id) << " ";
      } else {
        std::cout << "null ";
      }
    }
    printf("\n");
  }

 protected:
  virtual void AssertArraysSortToIndices(std::vector<std::string> values,
                                         const std::string& expected) {
    auto type = TypeTraits<ArrowType>::type_singleton();
    std::vector<std::shared_ptr<Array>> input;
    for (auto value : values) {
      input.push_back(ArrayFromJSON(type, value));
    }
    AssertSortArraysToIndices(input, ArrayFromJSON(uint64(), expected));
  }

  virtual void SortArraysToIndices(std::vector<std::string> values) {
    auto type = TypeTraits<ArrowType>::type_singleton();
    std::vector<std::shared_ptr<Array>> input;
    for (auto value : values) {
      input.push_back(ArrayFromJSON(type, value));
    }
    SortArraysToIndices(input);
  }
};

typedef testing::Types<Int32Type> IntegralArrowTypes;
template <typename ArrowType>
class TestSortToIndicesKernelForIntegral : public TestSortToIndicesKernel<ArrowType> {};
TYPED_TEST_CASE(TestSortToIndicesKernelForIntegral, IntegralArrowTypes);

TYPED_TEST(TestSortToIndicesKernelForIntegral, SortIntegral) {
  std::vector<std::string> input;
  input.push_back("[10, 12, 4, 50, 50, 32, 11]");
  input.push_back("[1, 14, 43, 42, 6, null, 2]");
  input.push_back("[3, 64, 15, 7, 9, 19, 33]");
  input.push_back("[23, 17, 41, 18, 20, 35, 30]");
  input.push_back("[37, null, 22, 13, 8, 59, 21]");
  this->SortArraysToIndices(input);
}

/*template <typename ArrowType>
class TestSortToIndicesKernelRandom : public ComputeFixture, public TestBase {};

using SortToIndicesableTypes =
    ::testing::Types<UInt8Type, UInt16Type, UInt32Type, UInt64Type, Int8Type, Int16Type,
                     Int32Type, Int64Type, FloatType, DoubleType, StringType>;

template <typename ArrayType>
class Comparator {
 public:
  bool operator()(const ArrayType& array, uint64_t lhs, uint64_t rhs) {
    if (array.IsNull(rhs) && array.IsNull(lhs)) return lhs < rhs;
    if (array.IsNull(rhs)) return true;
    if (array.IsNull(lhs)) return false;
    return array.Value(lhs) <= array.Value(rhs);
  }
};

template <>
class Comparator<StringArray> {
 public:
  bool operator()(const BinaryArray& array, uint64_t lhs, uint64_t rhs) {
    if (array.IsNull(rhs) && array.IsNull(lhs)) return lhs < rhs;
    if (array.IsNull(rhs)) return true;
    if (array.IsNull(lhs)) return false;
    return array.GetView(lhs) <= array.GetView(rhs);
  }
};

template <typename ArrayType>
void ValidateSorted(const ArrayType& array, UInt64Array& offsets) {
  Comparator<ArrayType> compare;
  for (int i = 1; i < array.length(); i++) {
    uint64_t lhs = offsets.Value(i - 1);
    uint64_t rhs = offsets.Value(i);
    ASSERT_TRUE(compare(array, lhs, rhs));
  }
}*/

}  // namespace compute
}  // namespace arrow
