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

#pragma once

#include <memory>

#include "arrow/compute/kernel.h"
#include "arrow/status.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;

namespace compute {

class FunctionContext;

struct ArrayItemIndex {
  uint64_t id = 0;
  uint64_t array_id = 0;
};

/// \brief Returns the indices that would sort an array.
///
/// Perform an indirect sort of array. The output array will contain
/// indices that would sort an array, which would be the same length
/// as input. Nulls will be stably partitioned to the end of the output.
///
/// For example given values = [null, 1, 3.3, null, 2, 5.3], the output
/// will be [1, 4, 2, 5, 0, 3]
///
/// \param[in] ctx the FunctionContext
/// \param[in] values arrays to sort
/// \param[out] offsets indices that would sort an array
ARROW_EXPORT
Status SortArraysToIndices(FunctionContext* ctx,
                           std::vector<std::shared_ptr<Array>> values,
                           std::shared_ptr<Array>* offsets, bool nulls_first, bool asc);

}  // namespace compute
}  // namespace arrow
