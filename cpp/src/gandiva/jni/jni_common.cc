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

#include <google/protobuf/io/coded_stream.h>

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include "gandiva/configuration.h"
#include "gandiva/filter.h"
#include "gandiva/jni/config_holder.h"
#include "gandiva/jni/env_helper.h"
#include "gandiva/jni/id_to_module_map.h"
#include "gandiva/jni/module_holder.h"
#include "gandiva/jni/protobuf_utils.h"
#include "gandiva/projector.h"
#include "gandiva/selection_vector.h"
#include "gandiva/tree_expr_builder.h"
#include "jni/org_apache_arrow_gandiva_evaluator_JniWrapper.h"

using gandiva::ConditionPtr;
using gandiva::DataTypePtr;
using gandiva::ExpressionPtr;
using gandiva::ExpressionVector;
using gandiva::FieldPtr;
using gandiva::FieldVector;
using gandiva::Filter;
using gandiva::NodePtr;
using gandiva::NodeVector;
using gandiva::Projector;
using gandiva::SchemaPtr;
using gandiva::Status;
using gandiva::TreeExprBuilder;

using gandiva::ArrayDataVector;
using gandiva::ConfigHolder;
using gandiva::Configuration;
using gandiva::ConfigurationBuilder;
using gandiva::FilterHolder;
using gandiva::ProjectorHolder;

// forward declarations
NodePtr ProtoTypeToNode(const types::TreeNode& node);

static jint JNI_VERSION = JNI_VERSION_1_6;

// extern refs - initialized for other modules.
jclass configuration_builder_class_;

// refs for self.
static jclass gandiva_exception_;
static jclass vector_expander_class_;
static jclass vector_expander_ret_class_;
static jmethodID vector_expander_method_;
static jfieldID vector_expander_ret_address_;
static jfieldID vector_expander_ret_capacity_;

// module maps
gandiva::IdToModuleMap<std::shared_ptr<ProjectorHolder>> projector_modules_;
gandiva::IdToModuleMap<std::shared_ptr<FilterHolder>> filter_modules_;

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }
  jclass local_configuration_builder_class_ =
      env->FindClass("org/apache/arrow/gandiva/evaluator/ConfigurationBuilder");
  configuration_builder_class_ =
      (jclass)env->NewGlobalRef(local_configuration_builder_class_);
  env->DeleteLocalRef(local_configuration_builder_class_);

  jclass localExceptionClass =
      env->FindClass("org/apache/arrow/gandiva/exceptions/GandivaException");
  gandiva_exception_ = (jclass)env->NewGlobalRef(localExceptionClass);
  env->ExceptionDescribe();
  env->DeleteLocalRef(localExceptionClass);

  jclass local_expander_class =
      env->FindClass("org/apache/arrow/gandiva/evaluator/VectorExpander");
  vector_expander_class_ = (jclass)env->NewGlobalRef(local_expander_class);
  env->DeleteLocalRef(local_expander_class);

  vector_expander_method_ = env->GetMethodID(
      vector_expander_class_, "expandOutputVectorAtIndex",
      "(II)Lorg/apache/arrow/gandiva/evaluator/VectorExpander$ExpandResult;");

  jclass local_expander_ret_class =
      env->FindClass("org/apache/arrow/gandiva/evaluator/VectorExpander$ExpandResult");
  vector_expander_ret_class_ = (jclass)env->NewGlobalRef(local_expander_ret_class);
  env->DeleteLocalRef(local_expander_ret_class);

  vector_expander_ret_address_ =
      env->GetFieldID(vector_expander_ret_class_, "address", "J");
  vector_expander_ret_capacity_ =
      env->GetFieldID(vector_expander_ret_class_, "capacity", "I");
  return JNI_VERSION;
}

void JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION);
  env->DeleteGlobalRef(configuration_builder_class_);
  env->DeleteGlobalRef(gandiva_exception_);
  env->DeleteGlobalRef(vector_expander_class_);
  env->DeleteGlobalRef(vector_expander_ret_class_);
}

// Common for both projector and filters.

Status make_record_batch_with_buf_addrs(SchemaPtr schema, int num_rows,
                                        jlong* in_buf_addrs, jlong* in_buf_sizes,
                                        int in_bufs_len,
                                        std::shared_ptr<arrow::RecordBatch>* batch) {
  std::vector<std::shared_ptr<arrow::ArrayData>> columns;
  auto num_fields = schema->num_fields();
  int buf_idx = 0;
  int sz_idx = 0;

  for (int i = 0; i < num_fields; i++) {
    auto field = schema->field(i);
    std::vector<std::shared_ptr<arrow::Buffer>> buffers;

    if (buf_idx >= in_bufs_len) {
      return Status::Invalid("insufficient number of in_buf_addrs");
    }
    jlong validity_addr = in_buf_addrs[buf_idx++];
    jlong validity_size = in_buf_sizes[sz_idx++];
    auto validity = std::shared_ptr<arrow::Buffer>(
        new arrow::Buffer(reinterpret_cast<uint8_t*>(validity_addr), validity_size));
    buffers.push_back(validity);

    if (buf_idx >= in_bufs_len) {
      return Status::Invalid("insufficient number of in_buf_addrs");
    }
    jlong value_addr = in_buf_addrs[buf_idx++];
    jlong value_size = in_buf_sizes[sz_idx++];
    auto data = std::shared_ptr<arrow::Buffer>(
        new arrow::Buffer(reinterpret_cast<uint8_t*>(value_addr), value_size));
    buffers.push_back(data);

    if (arrow::is_binary_like(field->type()->id())) {
      if (buf_idx >= in_bufs_len) {
        return Status::Invalid("insufficient number of in_buf_addrs");
      }

      // add offsets buffer for variable-len fields.
      jlong offsets_addr = in_buf_addrs[buf_idx++];
      jlong offsets_size = in_buf_sizes[sz_idx++];
      auto offsets = std::shared_ptr<arrow::Buffer>(
          new arrow::Buffer(reinterpret_cast<uint8_t*>(offsets_addr), offsets_size));
      buffers.push_back(offsets);
    }

    auto array_data = arrow::ArrayData::Make(field->type(), num_rows, std::move(buffers));
    columns.push_back(array_data);
  }
  *batch = arrow::RecordBatch::Make(schema, num_rows, columns);
  return Status::OK();
}

// projector related functions.
void releaseProjectorInput(jbyteArray schema_arr, jbyte* schema_bytes,
                           jbyteArray exprs_arr, jbyte* exprs_bytes, JNIEnv* env) {
  env->ReleaseByteArrayElements(schema_arr, schema_bytes, JNI_ABORT);
  env->ReleaseByteArrayElements(exprs_arr, exprs_bytes, JNI_ABORT);
}

JNIEXPORT jlong JNICALL Java_org_apache_arrow_gandiva_evaluator_JniWrapper_buildProjector(
    JNIEnv* env, jobject obj, jbyteArray schema_arr, jbyteArray exprs_arr,
    jint selection_vector_type, jlong configuration_id) {
  jlong module_id = 0LL;
  std::shared_ptr<Projector> projector;
  std::shared_ptr<ProjectorHolder> holder;

  types::Schema schema;
  jsize schema_len = env->GetArrayLength(schema_arr);
  jbyte* schema_bytes = env->GetByteArrayElements(schema_arr, 0);

  types::ExpressionList exprs;
  jsize exprs_len = env->GetArrayLength(exprs_arr);
  jbyte* exprs_bytes = env->GetByteArrayElements(exprs_arr, 0);

  ExpressionVector expr_vector;
  SchemaPtr schema_ptr;
  FieldVector ret_types;
  gandiva::Status status;
  auto mode = gandiva::SelectionVector::MODE_NONE;

  std::shared_ptr<Configuration> config = ConfigHolder::MapLookup(configuration_id);
  std::stringstream ss;

  if (config == nullptr) {
    ss << "configuration is mandatory.";
    releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
    goto err_out;
  }

  if (!ParseProtobuf(reinterpret_cast<uint8_t*>(schema_bytes), schema_len, &schema)) {
    ss << "Unable to parse schema protobuf\n";
    releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
    goto err_out;
  }

  if (!ParseProtobuf(reinterpret_cast<uint8_t*>(exprs_bytes), exprs_len, &exprs)) {
    releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
    ss << "Unable to parse expressions protobuf\n";
    goto err_out;
  }

  // convert types::Schema to arrow::Schema
  schema_ptr = ProtoTypeToSchema(schema);
  if (schema_ptr == nullptr) {
    ss << "Unable to construct arrow schema object from schema protobuf\n";
    releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
    goto err_out;
  }

  // create Expression out of the list of exprs
  for (int i = 0; i < exprs.exprs_size(); i++) {
    ExpressionPtr root = ProtoTypeToExpression(exprs.exprs(i));

    if (root == nullptr) {
      ss << "Unable to construct expression object from expression protobuf\n";
      releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
      goto err_out;
    }

    expr_vector.push_back(root);
    ret_types.push_back(root->result());
  }

  switch (selection_vector_type) {
    case types::SV_NONE:
      mode = gandiva::SelectionVector::MODE_NONE;
      break;
    case types::SV_INT16:
      mode = gandiva::SelectionVector::MODE_UINT16;
      break;
    case types::SV_INT32:
      mode = gandiva::SelectionVector::MODE_UINT32;
      break;
  }
  // good to invoke the evaluator now
  status = Projector::Make(schema_ptr, expr_vector, mode, config, &projector);

  if (!status.ok()) {
    ss << "Failed to make LLVM module due to " << status.message() << "\n";
    releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
    goto err_out;
  }

  // store the result in a map
  holder = std::shared_ptr<ProjectorHolder>(
      new ProjectorHolder(schema_ptr, ret_types, std::move(projector)));
  module_id = projector_modules_.Insert(holder);
  releaseProjectorInput(schema_arr, schema_bytes, exprs_arr, exprs_bytes, env);
  return module_id;

err_out:
  env->ThrowNew(gandiva_exception_, ss.str().c_str());
  return module_id;
}

///
/// \brief Resizable buffer which resizes by doing a callback into java.
///
class JavaResizableBuffer : public arrow::ResizableBuffer {
 public:
  JavaResizableBuffer(JNIEnv* env, jobject jexpander, int32_t vector_idx, uint8_t* buffer,
                      int32_t len)
      : ResizableBuffer(buffer, len),
        env_(env),
        jexpander_(jexpander),
        vector_idx_(vector_idx) {
    size_ = 0;
  }

  Status Resize(const int64_t new_size, bool shrink_to_fit) override;

  Status Reserve(const int64_t new_capacity) override {
    return Status::NotImplemented("reserve not implemented");
  }

 private:
  JNIEnv* env_;
  jobject jexpander_;
  int32_t vector_idx_;
};

Status JavaResizableBuffer::Resize(const int64_t new_size, bool shrink_to_fit) {
  if (shrink_to_fit == true) {
    return Status::NotImplemented("shrink not implemented");
  }

  if (ARROW_PREDICT_TRUE(new_size < capacity())) {
    // no need to expand.
    size_ = new_size;
    return Status::OK();
  }

  if (new_size > INT32_MAX) {
    return Status::OutOfMemory("java supports buffer sizes upto 2GB only");
  }

  // callback into java to expand the buffer
  int32_t updated_capacity = static_cast<int32_t>(new_size);
  jobject ret = env_->CallObjectMethod(jexpander_, vector_expander_method_, vector_idx_,
                                       updated_capacity);
  if (env_->ExceptionCheck()) {
    env_->ExceptionDescribe();
    env_->ExceptionClear();
    return Status::OutOfMemory("buffer expand failed in java");
  }

  jlong ret_address = env_->GetLongField(ret, vector_expander_ret_address_);
  jint ret_capacity = env_->GetIntField(ret, vector_expander_ret_capacity_);
  DCHECK_GE(ret_capacity, updated_capacity);

  data_ = mutable_data_ = reinterpret_cast<uint8_t*>(ret_address);
  size_ = new_size;
  capacity_ = ret_capacity;
  return Status::OK();
}

#define CHECK_OUT_BUFFER_IDX_AND_BREAK(idx, len)                               \
  if (idx >= len) {                                                            \
    status = gandiva::Status::Invalid("insufficient number of out_buf_addrs"); \
    break;                                                                     \
  }

JNIEXPORT void JNICALL
Java_org_apache_arrow_gandiva_evaluator_JniWrapper_evaluateProjector(
    JNIEnv* env, jobject object, jobject jexpander, jlong module_id, jint num_rows,
    jlongArray buf_addrs, jlongArray buf_sizes, jint sel_vec_type, jint sel_vec_rows,
    jlong sel_vec_addr, jlong sel_vec_size, jlongArray out_buf_addrs,
    jlongArray out_buf_sizes) {
  Status status;
  std::shared_ptr<ProjectorHolder> holder = projector_modules_.Lookup(module_id);
  if (holder == nullptr) {
    std::stringstream ss;
    ss << "Unknown module id " << module_id;
    env->ThrowNew(gandiva_exception_, ss.str().c_str());
    return;
  }

  int in_bufs_len = env->GetArrayLength(buf_addrs);
  if (in_bufs_len != env->GetArrayLength(buf_sizes)) {
    env->ThrowNew(gandiva_exception_, "mismatch in arraylen of buf_addrs and buf_sizes");
    return;
  }

  int out_bufs_len = env->GetArrayLength(out_buf_addrs);
  if (out_bufs_len != env->GetArrayLength(out_buf_sizes)) {
    env->ThrowNew(gandiva_exception_,
                  "mismatch in arraylen of out_buf_addrs and out_buf_sizes");
    return;
  }

  jlong* in_buf_addrs = env->GetLongArrayElements(buf_addrs, 0);
  jlong* in_buf_sizes = env->GetLongArrayElements(buf_sizes, 0);

  jlong* out_bufs = env->GetLongArrayElements(out_buf_addrs, 0);
  jlong* out_sizes = env->GetLongArrayElements(out_buf_sizes, 0);

  do {
    std::shared_ptr<arrow::RecordBatch> in_batch;
    status = make_record_batch_with_buf_addrs(holder->schema(), num_rows, in_buf_addrs,
                                              in_buf_sizes, in_bufs_len, &in_batch);
    if (!status.ok()) {
      break;
    }

    std::shared_ptr<gandiva::SelectionVector> selection_vector;
    auto selection_buffer = std::make_shared<arrow::Buffer>(
        reinterpret_cast<uint8_t*>(sel_vec_addr), sel_vec_size);
    int output_row_count = 0;
    switch (sel_vec_type) {
      case types::SV_NONE: {
        output_row_count = num_rows;
        break;
      }
      case types::SV_INT16: {
        status = gandiva::SelectionVector::MakeImmutableInt16(
            sel_vec_rows, selection_buffer, &selection_vector);
        output_row_count = sel_vec_rows;
        break;
      }
      case types::SV_INT32: {
        status = gandiva::SelectionVector::MakeImmutableInt32(
            sel_vec_rows, selection_buffer, &selection_vector);
        output_row_count = sel_vec_rows;
        break;
      }
    }
    if (!status.ok()) {
      break;
    }

    auto ret_types = holder->rettypes();
    ArrayDataVector output;
    int buf_idx = 0;
    int sz_idx = 0;
    int output_vector_idx = 0;
    for (FieldPtr field : ret_types) {
      std::vector<std::shared_ptr<arrow::Buffer>> buffers;

      CHECK_OUT_BUFFER_IDX_AND_BREAK(buf_idx, out_bufs_len);
      uint8_t* validity_buf = reinterpret_cast<uint8_t*>(out_bufs[buf_idx++]);
      jlong bitmap_sz = out_sizes[sz_idx++];
      buffers.push_back(std::make_shared<arrow::MutableBuffer>(validity_buf, bitmap_sz));

      if (arrow::is_binary_like(field->type()->id())) {
        CHECK_OUT_BUFFER_IDX_AND_BREAK(buf_idx, out_bufs_len);
        uint8_t* offsets_buf = reinterpret_cast<uint8_t*>(out_bufs[buf_idx++]);
        jlong offsets_sz = out_sizes[sz_idx++];
        buffers.push_back(
            std::make_shared<arrow::MutableBuffer>(offsets_buf, offsets_sz));
      }

      CHECK_OUT_BUFFER_IDX_AND_BREAK(buf_idx, out_bufs_len);
      uint8_t* value_buf = reinterpret_cast<uint8_t*>(out_bufs[buf_idx++]);
      jlong data_sz = out_sizes[sz_idx++];
      if (arrow::is_binary_like(field->type()->id())) {
        if (jexpander == nullptr) {
          status = Status::Invalid(
              "expression has variable len output columns, but the expander object is "
              "null");
          break;
        }
        buffers.push_back(std::make_shared<JavaResizableBuffer>(
            env, jexpander, output_vector_idx, value_buf, data_sz));
      } else {
        buffers.push_back(std::make_shared<arrow::MutableBuffer>(value_buf, data_sz));
      }

      auto array_data = arrow::ArrayData::Make(field->type(), output_row_count, buffers);
      output.push_back(array_data);
      ++output_vector_idx;
    }
    if (!status.ok()) {
      break;
    }
    status = holder->projector()->Evaluate(*in_batch, selection_vector.get(), output);
  } while (0);

  env->ReleaseLongArrayElements(buf_addrs, in_buf_addrs, JNI_ABORT);
  env->ReleaseLongArrayElements(buf_sizes, in_buf_sizes, JNI_ABORT);
  env->ReleaseLongArrayElements(out_buf_addrs, out_bufs, JNI_ABORT);
  env->ReleaseLongArrayElements(out_buf_sizes, out_sizes, JNI_ABORT);

  if (!status.ok()) {
    std::stringstream ss;
    ss << "Evaluate returned " << status.message() << "\n";
    env->ThrowNew(gandiva_exception_, status.message().c_str());
    return;
  }
}

JNIEXPORT void JNICALL Java_org_apache_arrow_gandiva_evaluator_JniWrapper_closeProjector(
    JNIEnv* env, jobject cls, jlong module_id) {
  projector_modules_.Erase(module_id);
}

// filter related functions.
void releaseFilterInput(jbyteArray schema_arr, jbyte* schema_bytes,
                        jbyteArray condition_arr, jbyte* condition_bytes, JNIEnv* env) {
  env->ReleaseByteArrayElements(schema_arr, schema_bytes, JNI_ABORT);
  env->ReleaseByteArrayElements(condition_arr, condition_bytes, JNI_ABORT);
}

JNIEXPORT jlong JNICALL Java_org_apache_arrow_gandiva_evaluator_JniWrapper_buildFilter(
    JNIEnv* env, jobject obj, jbyteArray schema_arr, jbyteArray condition_arr,
    jlong configuration_id) {
  jlong module_id = 0LL;
  std::shared_ptr<Filter> filter;
  std::shared_ptr<FilterHolder> holder;

  types::Schema schema;
  jsize schema_len = env->GetArrayLength(schema_arr);
  jbyte* schema_bytes = env->GetByteArrayElements(schema_arr, 0);

  types::Condition condition;
  jsize condition_len = env->GetArrayLength(condition_arr);
  jbyte* condition_bytes = env->GetByteArrayElements(condition_arr, 0);

  ConditionPtr condition_ptr;
  SchemaPtr schema_ptr;
  gandiva::Status status;

  std::shared_ptr<Configuration> config = ConfigHolder::MapLookup(configuration_id);
  std::stringstream ss;

  if (config == nullptr) {
    ss << "configuration is mandatory.";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  if (!ParseProtobuf(reinterpret_cast<uint8_t*>(schema_bytes), schema_len, &schema)) {
    ss << "Unable to parse schema protobuf\n";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  if (!ParseProtobuf(reinterpret_cast<uint8_t*>(condition_bytes), condition_len,
                     &condition)) {
    ss << "Unable to parse condition protobuf\n";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  // convert types::Schema to arrow::Schema
  schema_ptr = ProtoTypeToSchema(schema);
  if (schema_ptr == nullptr) {
    ss << "Unable to construct arrow schema object from schema protobuf\n";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  condition_ptr = ProtoTypeToCondition(condition);
  if (condition_ptr == nullptr) {
    ss << "Unable to construct condition object from condition protobuf\n";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  // good to invoke the filter builder now
  status = Filter::Make(schema_ptr, condition_ptr, config, &filter);
  if (!status.ok()) {
    ss << "Failed to make LLVM module due to " << status.message() << "\n";
    releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
    goto err_out;
  }

  // store the result in a map
  holder = std::shared_ptr<FilterHolder>(new FilterHolder(schema_ptr, std::move(filter)));
  module_id = filter_modules_.Insert(holder);
  releaseFilterInput(schema_arr, schema_bytes, condition_arr, condition_bytes, env);
  return module_id;

err_out:
  env->ThrowNew(gandiva_exception_, ss.str().c_str());
  return module_id;
}

JNIEXPORT jint JNICALL Java_org_apache_arrow_gandiva_evaluator_JniWrapper_evaluateFilter(
    JNIEnv* env, jobject cls, jlong module_id, jint num_rows, jlongArray buf_addrs,
    jlongArray buf_sizes, jint jselection_vector_type, jlong out_buf_addr,
    jlong out_buf_size) {
  gandiva::Status status;
  std::shared_ptr<FilterHolder> holder = filter_modules_.Lookup(module_id);
  if (holder == nullptr) {
    env->ThrowNew(gandiva_exception_, "Unknown module id\n");
    return -1;
  }

  int in_bufs_len = env->GetArrayLength(buf_addrs);
  if (in_bufs_len != env->GetArrayLength(buf_sizes)) {
    env->ThrowNew(gandiva_exception_, "mismatch in arraylen of buf_addrs and buf_sizes");
    return -1;
  }

  jlong* in_buf_addrs = env->GetLongArrayElements(buf_addrs, 0);
  jlong* in_buf_sizes = env->GetLongArrayElements(buf_sizes, 0);
  std::shared_ptr<gandiva::SelectionVector> selection_vector;

  do {
    std::shared_ptr<arrow::RecordBatch> in_batch;

    status = make_record_batch_with_buf_addrs(holder->schema(), num_rows, in_buf_addrs,
                                              in_buf_sizes, in_bufs_len, &in_batch);
    if (!status.ok()) {
      break;
    }

    auto selection_vector_type =
        static_cast<types::SelectionVectorType>(jselection_vector_type);
    auto out_buffer = std::make_shared<arrow::MutableBuffer>(
        reinterpret_cast<uint8_t*>(out_buf_addr), out_buf_size);
    switch (selection_vector_type) {
      case types::SV_INT16:
        status =
            gandiva::SelectionVector::MakeInt16(num_rows, out_buffer, &selection_vector);
        break;
      case types::SV_INT32:
        status =
            gandiva::SelectionVector::MakeInt32(num_rows, out_buffer, &selection_vector);
        break;
      default:
        status = gandiva::Status::Invalid("unknown selection vector type");
    }
    if (!status.ok()) {
      break;
    }

    status = holder->filter()->Evaluate(*in_batch, selection_vector);
  } while (0);

  env->ReleaseLongArrayElements(buf_addrs, in_buf_addrs, JNI_ABORT);
  env->ReleaseLongArrayElements(buf_sizes, in_buf_sizes, JNI_ABORT);

  if (!status.ok()) {
    std::stringstream ss;
    ss << "Evaluate returned " << status.message() << "\n";
    env->ThrowNew(gandiva_exception_, status.message().c_str());
    return -1;
  } else {
    int64_t num_slots = selection_vector->GetNumSlots();
    // Check integer overflow
    if (num_slots > INT_MAX) {
      std::stringstream ss;
      ss << "The selection vector has " << num_slots
         << " slots, which is larger than the " << INT_MAX << " limit.\n";
      const std::string message = ss.str();
      env->ThrowNew(gandiva_exception_, message.c_str());
      return -1;
    }
    return static_cast<int>(num_slots);
  }
}

JNIEXPORT void JNICALL Java_org_apache_arrow_gandiva_evaluator_JniWrapper_closeFilter(
    JNIEnv* env, jobject cls, jlong module_id) {
  filter_modules_.Erase(module_id);
}
