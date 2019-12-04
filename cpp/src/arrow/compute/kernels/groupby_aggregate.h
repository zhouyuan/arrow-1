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

#ifndef ARROW_COMPUTE_KERNELS_GROUP_H
#define ARROW_COMPUTE_KERNELS_GROUP_H

#include <memory>

#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/util_internal.h"
#include "arrow/status.h"
#include "arrow/util/hashing.h"
#include "arrow/util/visibility.h"

namespace arrow {

class Array;
class DataType;
struct ArrayData;

namespace compute {

class FunctionContext;

/// \brief Compute group_id of elements from an array-like object
///
/// Note if a null occurs in the input it will NOT be included in the output.
///
/// \param[in] context the FunctionContext
/// \param[in] datum array-like input
/// \param[in] memo_table hashtable input
/// \param[out] out result as Array
template <typename InType, typename MemoTableType>
ARROW_EXPORT Status Group(FunctionContext* context, const Datum& datum,
                          std::shared_ptr<MemoTableType> memo_table,
                          std::shared_ptr<Array>* out);

/// \param[in] context the FunctionContext
/// \param[in] datum array-like input
/// \param[out] out result as Array
ARROW_EXPORT Status Group(FunctionContext* context, const Datum& datum,
                          std::shared_ptr<Array>* out);

}  // namespace compute
}  // namespace arrow
#endif
