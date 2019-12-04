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

#include "arrow/compute/kernels/hash.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "arrow/array.h"
#include "arrow/array/dict_internal.h"
#include "arrow/buffer.h"
#include "arrow/builder.h"
#include "arrow/compute/context.h"
#include "arrow/compute/kernel.h"
#include "arrow/compute/kernels/util_internal.h"
#include "arrow/type.h"
#include "arrow/type_traits.h"
#include "arrow/util/checked_cast.h"
#include "arrow/util/hashing.h"
#include "arrow/util/logging.h"
#include "arrow/util/macros.h"
#include "arrow/util/string_view.h"
#include "arrow/visitor_inline.h"

namespace arrow {

class MemoryPool;

using internal::checked_cast;
using internal::DictionaryTraits;
using internal::HashTraits;

namespace compute {

#define CHECK_IMPLEMENTED(KERNEL, FUNCNAME, TYPE)                                       \
  if (!KERNEL) {                                                                        \
    return Status::NotImplemented(FUNCNAME, " not implemented for ", type->ToString()); \
  }

class ActionBase {
 public:
  ActionBase(const std::shared_ptr<DataType>& type, MemoryPool* pool)
      : type_(type), pool_(pool) {}

 protected:
  std::shared_ptr<DataType> type_;
  MemoryPool* pool_;
};

class GroupAction final : public ActionBase {
 public:
  using ActionBase::ActionBase;

  GroupAction(const std::shared_ptr<DataType>& type, MemoryPool* pool)
      : ActionBase(type, pool), indices_builder_(pool) {}

  Status Reset() {
    indices_builder_.Reset();
    return Status::OK();
  }

  Status Reserve(const int64_t length) { return indices_builder_.Reserve(length); }

  template <class Index>
  void ObserveNullFound(Index index) {
    indices_builder_.UnsafeAppendNull();
  }

  template <class Index>
  void ObserveNullNotFound(Index index) {
    indices_builder_.UnsafeAppendNull();
  }

  template <class Index>
  void ObserveFound(Index index) {
    indices_builder_.UnsafeAppend(index);
  }

  template <class Index>
  void ObserveNotFound(Index index) {
    ObserveFound(index);
  }

  Status Flush(Datum* out) {
    std::shared_ptr<ArrayData> result;
    RETURN_NOT_OK(indices_builder_.FinishInternal(&result));
    out->value = std::move(result);
    return Status::OK();
  }

  std::shared_ptr<DataType> out_type() const { return int32(); }
  Status FlushFinal(Datum* out) { return Status::OK(); }

 private:
  Int32Builder indices_builder_;
};

// ----------------------------------------------------------------------
// Base class for all hash kernel implementations

class HashKernel : public UnaryKernel {
 public:
  // Reset for another run.
  virtual Status Reset() = 0;
  // Prepare the Action for the given input (e.g. reserve appropriately sized
  // data structures) and visit the given input with Action.
  virtual Status Append(FunctionContext* ctx, const ArrayData& input) = 0;
  // Flush out accumulated results from the last invocation of Call.
  virtual Status Flush(Datum* out) = 0;
  // Flush out accumulated results across all invocations of Call. The kernel
  // should not be used until after Reset() is called.
  virtual Status FlushFinal(Datum* out) = 0;
  // Get the values (keys) acummulated in the dictionary so far.
  virtual Status GetDictionary(std::shared_ptr<ArrayData>* out) = 0;
};

class HashKernelImpl : public HashKernel {
 public:
  Status Call(FunctionContext* ctx, const Datum& input, Datum* out) override {
    DCHECK_EQ(Datum::ARRAY, input.kind());
    RETURN_NOT_OK(Append(ctx, *input.array()));
    return Flush(out);
  }

  Status Append(FunctionContext* ctx, const ArrayData& input) override {
    std::lock_guard<std::mutex> guard(lock_);
    return Append(input);
  }

  virtual Status Append(const ArrayData& arr) = 0;

 protected:
  std::mutex lock_;
};

// ----------------------------------------------------------------------
// Base class for all "regular" hash kernel implementations
// (NullType has a separate implementation)

template <typename Type, typename Scalar, typename Action, bool with_error_status = false,
          bool with_memo_visit_null = true,
          typename MemoTableType = std::shared_ptr<internal::MemoTable>>
class RegularHashKernelImpl : public HashKernelImpl {
 public:
  RegularHashKernelImpl(const std::shared_ptr<DataType>& type, MemoryPool* pool,
                        std::shared_ptr<MemoTableType> memo_table)
      : pool_(pool), type_(type), action_(type, pool) {
    memo_table_ = memo_table;
  }

  Status Reset() override {
    // memo_table_.reset(new MemoTableType(pool_, 0));
    return action_.Reset();
  }

  Status Append(const ArrayData& arr) override {
    RETURN_NOT_OK(action_.Reserve(arr.length));
    return ArrayDataVisitor<Type>::Visit(arr, this);
  }

  Status Flush(Datum* out) override { return action_.Flush(out); }

  Status FlushFinal(Datum* out) override { return action_.FlushFinal(out); }

  Status GetDictionary(std::shared_ptr<ArrayData>* out) override {
    return DictionaryTraits<Type>::GetDictionaryArrayData(
        pool_, type_, *memo_table_.get(), 0 /* start_offset */, out);
  }

  template <bool HasError = with_error_status>
  enable_if_t<!HasError, Status> VisitNull() {
    auto on_found = [this](int32_t memo_index) { action_.ObserveNullFound(memo_index); };
    auto on_not_found = [this](int32_t memo_index) {
      action_.ObserveNullNotFound(memo_index);
    };

    if (with_memo_visit_null) {
      memo_table_->GetOrInsertNull(on_found, on_not_found);
    } else {
      action_.ObserveNullNotFound(-1);
    }
    return Status::OK();
  }

  template <bool HasError = with_error_status>
  enable_if_t<HasError, Status> VisitNull() {
    Status s = Status::OK();
    auto on_found = [this](int32_t memo_index) { action_.ObserveFound(memo_index); };
    auto on_not_found = [this, &s](int32_t memo_index) {
      action_.ObserveNotFound(memo_index, &s);
    };

    if (with_memo_visit_null) {
      memo_table_->GetOrInsertNull(on_found, on_not_found);
    } else {
      action_.ObserveNullNotFound(-1);
    }

    return s;
  }

  template <bool HasError = with_error_status>
  enable_if_t<!HasError, Status> VisitValue(const Scalar& value) {
    auto on_found = [this](int32_t memo_index) { action_.ObserveFound(memo_index); };
    auto on_not_found = [this](int32_t memo_index) {
      action_.ObserveNotFound(memo_index);
    };

    memo_table_->GetOrInsert(value, on_found, on_not_found);
    return Status::OK();
  }

  template <bool HasError = with_error_status>
  enable_if_t<HasError, Status> VisitValue(const Scalar& value) {
    Status s = Status::OK();
    auto on_found = [this](int32_t memo_index) { action_.ObserveFound(memo_index); };
    auto on_not_found = [this, &s](int32_t memo_index) {
      action_.ObserveNotFound(memo_index, &s);
    };
    memo_table_->GetOrInsert(value, on_found, on_not_found);
    return s;
  }

  std::shared_ptr<DataType> out_type() const override { return action_.out_type(); }

 protected:
  MemoryPool* pool_;
  std::shared_ptr<DataType> type_;
  Action action_;
  std::shared_ptr<MemoTableType> memo_table_;
};

// ----------------------------------------------------------------------
// Hash kernel implementation for nulls

template <typename Action, typename MemoTableType>
class NullHashKernelImpl : public HashKernelImpl {
 public:
  NullHashKernelImpl(const std::shared_ptr<DataType>& type, MemoryPool* pool,
                     std::shared_ptr<MemoTableType> memo_table)
      : pool_(pool), type_(type), action_(type, pool) {}

  Status Reset() override { return action_.Reset(); }

  Status Append(const ArrayData& arr) override {
    RETURN_NOT_OK(action_.Reserve(arr.length));
    for (int64_t i = 0; i < arr.length; ++i) {
      if (i == 0) {
        action_.ObserveNullNotFound(0);
      } else {
        action_.ObserveNullFound(0);
      }
    }
    return Status::OK();
  }

  Status Flush(Datum* out) override { return action_.Flush(out); }
  Status FlushFinal(Datum* out) override { return action_.FlushFinal(out); }

  Status GetDictionary(std::shared_ptr<ArrayData>* out) override {
    // TODO(wesm): handle null being a valid dictionary value
    auto null_array = std::make_shared<NullArray>(0);
    *out = null_array->data();
    return Status::OK();
  }

  std::shared_ptr<DataType> out_type() const override { return null(); }

 protected:
  MemoryPool* pool_;
  std::shared_ptr<DataType> type_;
  Action action_;
};

// ----------------------------------------------------------------------
// Kernel wrapper for generic hash table kernels

template <typename Type, typename Action, bool with_error_status,
          bool with_memo_visit_null, typename MemoTableType, typename Enable = void>
struct HashKernelTraits {};

template <typename Type, typename Action, bool with_error_status,
          bool with_memo_visit_null, typename MemoTableType>
struct HashKernelTraits<Type, Action, with_error_status, with_memo_visit_null,
                        MemoTableType, enable_if_null<Type>> {
  using HashKernelImpl = NullHashKernelImpl<Action, MemoTableType>;
};

template <typename Type, typename Action, bool with_error_status,
          bool with_memo_visit_null, typename MemoTableType>
struct HashKernelTraits<Type, Action, with_error_status, with_memo_visit_null,
                        MemoTableType, enable_if_has_c_type<Type>> {
  using HashKernelImpl =
      RegularHashKernelImpl<Type, typename Type::c_type, Action, with_error_status,
                            with_memo_visit_null, MemoTableType>;
};

template <typename Type, typename Action, bool with_error_status,
          bool with_memo_visit_null, typename MemoTableType>
struct HashKernelTraits<Type, Action, with_error_status, with_memo_visit_null,
                        MemoTableType, enable_if_has_string_view<Type>> {
  using HashKernelImpl =
      RegularHashKernelImpl<Type, util::string_view, Action, with_error_status,
                            with_memo_visit_null, MemoTableType>;
};

template <typename InType, typename MemoTableType>
Status GetGroupKernel(FunctionContext* ctx, const std::shared_ptr<DataType>& type,
                      std::shared_ptr<MemoTableType> memo_table,
                      std::unique_ptr<HashKernel>* out) {
  std::unique_ptr<HashKernel> kernel;
  kernel.reset(
      new
      typename HashKernelTraits<InType, GroupAction, false, false,
                                MemoTableType>::HashKernelImpl(type, ctx->memory_pool(),
                                                               memo_table));

  CHECK_IMPLEMENTED(kernel, "group", type);
  RETURN_NOT_OK(kernel->Reset());
  *out = std::move(kernel);
  return Status::OK();
}

template <typename InType, typename MemoTableType>
Status Group(FunctionContext* ctx, const Datum& value,
             std::shared_ptr<MemoTableType> memo_table, std::shared_ptr<Array>* out) {
  std::unique_ptr<HashKernel> func;
  RETURN_NOT_OK(GetGroupKernel<InType>(ctx, value.type(), memo_table, &func));

  std::vector<Datum> indices_outputs;
  RETURN_NOT_OK(detail::InvokeUnaryArrayKernel(ctx, func.get(), value, &indices_outputs));

  if (indices_outputs.size() == 0) {
    RETURN_NOT_OK(
        MakeArrayOfNull(arrow::TypeTraits<UInt32Type>::type_singleton(), 0, out));
  } else {
    // Create the dictionary type
    DCHECK_EQ(indices_outputs[0].kind(), Datum::ARRAY);

    for (const Datum& datum : indices_outputs) {
      *out = MakeArray(datum.array());
      break;
    }
  }
  return Status::OK();
}

#define PROCESS_SUPPORTED_TYPES(PROCESS) \
  PROCESS(BooleanType)                   \
  PROCESS(UInt8Type)                     \
  PROCESS(Int8Type)                      \
  PROCESS(UInt16Type)                    \
  PROCESS(Int16Type)                     \
  PROCESS(UInt32Type)                    \
  PROCESS(Int32Type)                     \
  PROCESS(UInt64Type)                    \
  PROCESS(Int64Type)                     \
  PROCESS(FloatType)                     \
  PROCESS(DoubleType)                    \
  PROCESS(Date32Type)                    \
  PROCESS(Date64Type)                    \
  PROCESS(Time32Type)                    \
  PROCESS(Time64Type)                    \
  PROCESS(TimestampType)                 \
  PROCESS(BinaryType)                    \
  PROCESS(StringType)                    \
  PROCESS(FixedSizeBinaryType)           \
  PROCESS(Decimal128Type)

Status Group(FunctionContext* ctx, const Datum& value, std::shared_ptr<Array>* out) {
  switch (value.type()->id()) {
#define PROCESS(InType)                                                                \
  case InType::type_id: {                                                              \
    using MemoTableType = typename arrow::internal::HashTraits<InType>::MemoTableType; \
    auto memo_table = std::make_shared<MemoTableType>(ctx->memory_pool());             \
    return Group<InType>(ctx, value, memo_table, out);                                 \
  } break;
    PROCESS_SUPPORTED_TYPES(PROCESS)
#undef PROCESS
    case NullType::type_id: {
      std::shared_ptr<internal::MemoTable> memo_table;
      return Group<NullType>(ctx, value, memo_table, out);
    } break;
    default:
      break;
  }
  return arrow::Status::OK();
}
#undef PROCESS_SUPPORTED_TYPES

}  // namespace compute
}  // namespace arrow
