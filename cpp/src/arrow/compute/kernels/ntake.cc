// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// returnGegarding copyright ownership.  The ASF licenses this file
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
#include <utility>
#include <vector>

#include "arrow/array/concatenate.h"
#include "arrow/compute/kernels/ntake.h"
#include "arrow/compute/kernels/ntake_internal.h"
#include "arrow/util/logging.h"
#include "arrow/visitor_inline.h"

namespace arrow {
namespace compute {

template <typename IndexType>
class NTakeKernelImpl : public NTakeKernel {
 public:
  explicit NTakeKernelImpl(const std::shared_ptr<DataType>& value_type)
      : NTakeKernel(value_type) {}

  Status Init() {
    return NTaker<ArrayIndexSequence<IndexType>>::Make(this->type_, &taker_);
  }

  Status NTake(FunctionContext* ctx, const Array& values, const Array& indices_array,
              std::shared_ptr<Array>* out) override {
    RETURN_NOT_OK(taker_->SetContext(ctx));
    RETURN_NOT_OK(taker_->Take(values, ArrayIndexSequence<IndexType>(indices_array)));
    return taker_->Finish(out);
  }

  std::unique_ptr<NTaker<ArrayIndexSequence<IndexType>>> taker_;
};

struct NUnpackIndices {
  template <typename IndexType>
  enable_if_integer<IndexType, Status> Visit(const IndexType&) {
    auto out = new NTakeKernelImpl<IndexType>(value_type_);
    out_->reset(out);
    return out->Init();
  }

  Status Visit(const DataType& other) {
    return Status::TypeError("index type not supported: ", other);
  }

  std::shared_ptr<DataType> value_type_;
  std::unique_ptr<NTakeKernel>* out_;
};

Status NTakeKernel::Make(const std::shared_ptr<DataType>& value_type,
                        const std::shared_ptr<DataType>& index_type,
                        std::unique_ptr<NTakeKernel>* out) {
  NUnpackIndices visitor{value_type, out};
  return VisitTypeInline(*index_type, &visitor);
}

Status NTakeKernel::Call(FunctionContext* ctx, const Datum& values, const Datum& indices,
                        Datum* out) {
  if (!values.is_array() || !indices.is_array()) {
    return Status::Invalid("NTakeKernel expects array values and indices");
  }
  auto values_array = values.make_array();
  auto indices_array = indices.make_array();
  std::shared_ptr<Array> out_array;
  RETURN_NOT_OK(NTake(ctx, *values_array, *indices_array, &out_array));
  *out = Datum(out_array);
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const Array& values, const Array& indices,
            const NTakeOptions& options, std::shared_ptr<Array>* out) {
  Datum out_datum;
  RETURN_NOT_OK(
      NTake(ctx, Datum(values.data()), Datum(indices.data()), options, &out_datum));
  *out = out_datum.make_array();
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const Datum& values, const Datum& indices,
            const NTakeOptions& options, Datum* out) {
  std::unique_ptr<NTakeKernel> kernel;
  RETURN_NOT_OK(NTakeKernel::Make(values.type(), indices.type(), &kernel));
  return kernel->Call(ctx, values, indices, out);
}

Status NTake(FunctionContext* ctx, const ChunkedArray& values, const Array& indices,
            const NTakeOptions& options, std::shared_ptr<ChunkedArray>* out) {
  auto num_chunks = values.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(1);  // Hard-coded 1 for now
  std::shared_ptr<Array> current_chunk;

  // Case 1: `values` has a single chunk, so just use it
  if (num_chunks == 1) {
    current_chunk = values.chunk(0);
  } else {
    // TODO Case 2: See if all `indices` fall in the same chunk and call Array Take on it
    // See
    // https://github.com/apache/arrow/blob/6f2c9041137001f7a9212f244b51bc004efc29af/r/src/compute.cpp#L123-L151
    // TODO Case 3: If indices are sorted, can slice them and call Array Take

    // Case 4: Else, concatenate chunks and call Array Take
    RETURN_NOT_OK(Concatenate(values.chunks(), default_memory_pool(), &current_chunk));
  }
  // Call Array Take on our single chunk
  RETURN_NOT_OK(NTake(ctx, *current_chunk, indices, options, &new_chunks[0]));
  *out = std::make_shared<ChunkedArray>(std::move(new_chunks));
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const ChunkedArray& values, const ChunkedArray& indices,
            const NTakeOptions& options, std::shared_ptr<ChunkedArray>* out) {
  auto num_chunks = indices.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(num_chunks);
  std::shared_ptr<ChunkedArray> current_chunk;

  for (int i = 0; i < num_chunks; i++) {
    // Take with that indices chunk
    // Note that as currently implemented, this is inefficient because `values`
    // will get concatenated on every iteration of this loop
    RETURN_NOT_OK(NTake(ctx, values, *indices.chunk(i), options, &current_chunk));
    // Concatenate the result to make a single array for this chunk
    RETURN_NOT_OK(
        Concatenate(current_chunk->chunks(), default_memory_pool(), &new_chunks[i]));
  }
  *out = std::make_shared<ChunkedArray>(std::move(new_chunks));
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const Array& values, const ChunkedArray& indices,
            const NTakeOptions& options, std::shared_ptr<ChunkedArray>* out) {
  auto num_chunks = indices.num_chunks();
  std::vector<std::shared_ptr<Array>> new_chunks(num_chunks);

  for (int i = 0; i < num_chunks; i++) {
    // Take with that indices chunk
    RETURN_NOT_OK(NTake(ctx, values, *indices.chunk(i), options, &new_chunks[i]));
  }
  *out = std::make_shared<ChunkedArray>(std::move(new_chunks));
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const RecordBatch& batch, const Array& indices,
            const NTakeOptions& options, std::shared_ptr<RecordBatch>* out) {
  auto ncols = batch.num_columns();
  auto nrows = indices.length();

  std::vector<std::shared_ptr<Array>> columns(ncols);

  for (int j = 0; j < ncols; j++) {
    RETURN_NOT_OK(NTake(ctx, *batch.column(j), indices, options, &columns[j]));
  }
  *out = RecordBatch::Make(batch.schema(), nrows, columns);
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const Table& table, const Array& indices,
            const NTakeOptions& options, std::shared_ptr<Table>* out) {
  auto ncols = table.num_columns();
  std::vector<std::shared_ptr<ChunkedArray>> columns(ncols);

  for (int j = 0; j < ncols; j++) {
    RETURN_NOT_OK(NTake(ctx, *table.column(j), indices, options, &columns[j]));
  }
  *out = Table::Make(table.schema(), columns);
  return Status::OK();
}

Status NTake(FunctionContext* ctx, const Table& table, const ChunkedArray& indices,
            const NTakeOptions& options, std::shared_ptr<Table>* out) {
  auto ncols = table.num_columns();
  std::vector<std::shared_ptr<ChunkedArray>> columns(ncols);

  for (int j = 0; j < ncols; j++) {
    RETURN_NOT_OK(NTake(ctx, *table.column(j), indices, options, &columns[j]));
  }
  *out = Table::Make(table.schema(), columns);
  return Status::OK();
}

}  // namespace compute
}  // namespace arrow
