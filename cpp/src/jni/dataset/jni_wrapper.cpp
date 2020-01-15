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

#include <arrow/dataset/file_base.h>
#include <arrow/dataset/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/ipc/api.h>
#include <arrow/util/iterator.h>
#include <arrow/filesystem/hdfs.h>
#include <arrow/io/api.h>
#include "jni/concurrent_map.h"

#include "org_apache_arrow_dataset_file_JniWrapper.h"
#include "org_apache_arrow_dataset_jni_JniWrapper.h"

static jclass illegal_access_exception_class;
static jclass illegal_argument_exception_class;
static jclass runtime_exception_class;

static jclass record_batch_handle_class;
static jclass record_batch_handle_field_class;
static jclass record_batch_handle_buffer_class;

static jmethodID record_batch_handle_constructor;
static jmethodID record_batch_handle_field_constructor;
static jmethodID record_batch_handle_buffer_constructor;

static jint JNI_VERSION = JNI_VERSION_1_6;

using arrow::jni::ConcurrentMap;

static ConcurrentMap<arrow::dataset::DataSourceDiscoveryPtr > data_source_discovery_holder_;
static ConcurrentMap<arrow::dataset::DataSourcePtr> data_source_holder_;
static ConcurrentMap<arrow::dataset::DataFragmentPtr > data_fragment_holder_;
static ConcurrentMap<arrow::dataset::ScanTaskPtr> scan_task_holder_;
static ConcurrentMap<std::shared_ptr<arrow::dataset::Scanner>> scanner_holder_;
static ConcurrentMap<std::shared_ptr<arrow::RecordBatchIterator>> iterator_holder_;
static ConcurrentMap<std::shared_ptr<arrow::Buffer>> buffer_holder_;

jclass CreateGlobalClassReference(JNIEnv* env, const char* class_name) {
  jclass local_class = env->FindClass(class_name);
  jclass global_class = (jclass)env->NewGlobalRef(local_class);
  env->DeleteLocalRef(local_class);
  return global_class;
}

jmethodID GetMethodID(JNIEnv* env, jclass this_class, const char* name, const char* sig) {
  jmethodID ret = env->GetMethodID(this_class, name, sig);
  if (ret == nullptr) {
    std::string error_message = "Unable to find method " + std::string(name) +
        " within signature" + std::string(sig);
    env->ThrowNew(illegal_access_exception_class, error_message.c_str());
  }
  return ret;
}

jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION) != JNI_OK) {
    return JNI_ERR;
  }

  illegal_access_exception_class =
      CreateGlobalClassReference(env, "Ljava/lang/IllegalAccessException;");
  illegal_argument_exception_class =
      CreateGlobalClassReference(env, "Ljava/lang/IllegalArgumentException;");
  runtime_exception_class =
      CreateGlobalClassReference(env, "Ljava/lang/RuntimeException;");

  record_batch_handle_class =
      CreateGlobalClassReference(env, "Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle;");
  record_batch_handle_field_class =
      CreateGlobalClassReference(env, "Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle$Field;");
  record_batch_handle_buffer_class =
      CreateGlobalClassReference(env, "Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle$Buffer;");

  record_batch_handle_constructor = GetMethodID(env, record_batch_handle_class, "<init>",
                                                "(J[Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle$Field;"
                                                "[Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle$Buffer;)V");
  record_batch_handle_field_constructor = GetMethodID(env, record_batch_handle_field_class, "<init>",
                                                      "(JJ)V");
  record_batch_handle_buffer_constructor = GetMethodID(env, record_batch_handle_buffer_class, "<init>",
                                                       "(JJJJ)V");

  env->ExceptionDescribe();

  return JNI_VERSION;
}


void JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION);
  env->DeleteGlobalRef(illegal_access_exception_class);
  env->DeleteGlobalRef(illegal_argument_exception_class);
  env->DeleteGlobalRef(runtime_exception_class);
  env->DeleteGlobalRef(record_batch_handle_class);
  env->DeleteGlobalRef(record_batch_handle_field_class);
  env->DeleteGlobalRef(record_batch_handle_buffer_class);

  data_source_discovery_holder_.Clear();
  data_source_holder_.Clear();
  data_fragment_holder_.Clear();
  scan_task_holder_.Clear();
  scanner_holder_.Clear();
  iterator_holder_.Clear();
}

std::shared_ptr<arrow::Schema> SchemaFromColumnNames(
    const std::shared_ptr<arrow::Schema>& input, const std::vector<std::string>& column_names) {
  std::vector<std::shared_ptr<arrow::Field>> columns;
  for (const auto& name : column_names) {
    columns.push_back(input->GetFieldByName(name));
  }
  return std::make_shared<arrow::Schema>(columns);
}

std::shared_ptr<arrow::dataset::FileFormat> GetFileFormat(JNIEnv *env, jint id) {
  switch (id) {
    case 0:
      return std::make_shared<arrow::dataset::ParquetFileFormat>();
    default:
      std::string error_message = "illegal file format id: " + std::to_string(id);
      env->ThrowNew(illegal_argument_exception_class, error_message.c_str());
      return nullptr; // unreachable
  }
}

arrow::fs::FileSystemPtr GetFileSystem(JNIEnv *env, jint id, std::string path,
                                       std::string* out_path) {
  switch (id) {
    case 0:
      *out_path = path;
      return std::make_shared<arrow::fs::LocalFileSystem>();
    case 1: {
      arrow::fs::FileSystemPtr ret;
      arrow::fs::FileSystemFromUri(path, &ret, out_path).ok();
      return ret;
    }
    default:std::string error_message = "illegal filesystem id: " + std::to_string(id);
      env->ThrowNew(illegal_argument_exception_class, error_message.c_str());
      return nullptr; // unreachable
  }
}

std::string JStringToCString(JNIEnv* env, jstring string) {
  jboolean copied;
  int32_t length = env->GetStringUTFLength(string);
  const char *chars = env->GetStringUTFChars(string, &copied);
  std::string str = std::string(chars, length);
  // fixme calling ReleaseStringUTFChars if memory leak faced
  return str;
}

std::vector<std::string> ToStringVector(JNIEnv* env, jobjectArray& str_array) {
  int length = env->GetArrayLength(str_array);
  std::vector<std::string> vector;
  for (int i = 0; i < length; i++) {
    auto string = (jstring) (env->GetObjectArrayElement(str_array, i));
    vector.push_back(JStringToCString(env, string));
  }
  return vector;
}

template <typename T>
std::vector<T> collect(arrow::Iterator<T> itr) {
  std::vector<T> vector;
  T t;
  while(true) {
    auto status = itr.Next(&t);
    if (!status.ok()) {
      return std::vector<T>(); // fixme
    }
    if (!t) {
      break;
    }
    vector.push_back(t);
  }
  return vector;
}

// FIXME: COPIED FROM intel/master on which this branch is not rebased yet
// FIXME: https://github.com/Intel-bigdata/arrow/blob/02502a4eb59834c2471dd629e77dbeed19559f68/cpp/src/jni/jni_common.h#L239-L254
jbyteArray ToSchemaByteArray(JNIEnv* env, std::shared_ptr<arrow::Schema> schema) {
  arrow::Status status;
  std::shared_ptr<arrow::Buffer> buffer;
  status = arrow::ipc::SerializeSchema(*schema.get(), nullptr,
                                       arrow::default_memory_pool(), &buffer);
  if (!status.ok()) {
    std::string error_message =
        "Unable to convert schema to byte array, err is " + status.message();
    env->ThrowNew(runtime_exception_class, error_message.c_str());
  }

  jbyteArray out = env->NewByteArray(buffer->size());
  auto src = reinterpret_cast<const jbyte*>(buffer->data());
  env->SetByteArrayRegion(out, 0, buffer->size(), src);
  return out;
}

// FIXME: COPIED FROM intel/master on which this branch is not rebased yet
// FIXME: https://github.com/Intel-bigdata/arrow/blob/02502a4eb59834c2471dd629e77dbeed19559f68/cpp/src/jni/jni_common.h#L256-L272
arrow::Status FromSchemaByteArray(JNIEnv* env, jbyteArray schemaBytes,
                                  std::shared_ptr<arrow::Schema>* schema) {
  arrow::Status status;
  arrow::ipc::DictionaryMemo in_memo;

  int schemaBytes_len = env->GetArrayLength(schemaBytes);
  jbyte* schemaBytes_data = env->GetByteArrayElements(schemaBytes, 0);

  auto serialized_schema =
      std::make_shared<arrow::Buffer>((uint8_t*)schemaBytes_data, schemaBytes_len);
  arrow::io::BufferReader buf_reader(serialized_schema);
  status = arrow::ipc::ReadSchema(&buf_reader, &in_memo, schema);

  env->ReleaseByteArrayElements(schemaBytes, schemaBytes_data, JNI_ABORT);

  return status;
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeDataSourceDiscovery
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeDataSourceDiscovery
    (JNIEnv *, jobject, jlong id) {
  data_source_discovery_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    inspectSchema
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_inspectSchema
    (JNIEnv* env, jobject, jlong data_source_discovery_id) {
  std::shared_ptr<arrow::dataset::DataSourceDiscovery> d
      = data_source_discovery_holder_.Lookup(data_source_discovery_id);
  std::shared_ptr<arrow::Schema> schema = d->Inspect().ValueOrDie();// fixme ValueOrDie
  return ToSchemaByteArray(env, schema);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    createDataSource
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_createDataSource
    (JNIEnv *, jobject, jlong data_source_discovery_id) {
  std::shared_ptr<arrow::dataset::DataSourceDiscovery> d
      = data_source_discovery_holder_.Lookup(data_source_discovery_id);
  arrow::dataset::DataSourcePtr data_source = d->Finish().ValueOrDie();// fixme ValueOrDie
  return data_source_holder_.Insert(data_source);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeDataSource
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeDataSource
    (JNIEnv *, jobject, jlong id) {
  data_source_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    getFragments
 * Signature: (J[Ljava/lang/String;[B)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_getFragments
    (JNIEnv* env, jobject, jlong data_source_id, jlong batch_size) {
  // todo consider filter buffer (the last param) which is currently ignored
  arrow::dataset::DataSourcePtr data_source = data_source_holder_.Lookup(data_source_id);
  arrow::dataset::ScanOptionsPtr scan_options = arrow::dataset::ScanOptions::Defaults();
  scan_options->batch_size = batch_size;
  arrow::dataset::DataFragmentIterator itr = data_source->GetFragments(scan_options); // todo consider column names projector (and output schema should be kept up with)
  std::vector<arrow::dataset::DataFragmentPtr> vector = collect(std::move(itr));
  jlongArray ret = env->NewLongArray(vector.size());
  for (unsigned long i = 0; i < vector.size(); i++) {
    arrow::dataset::DataFragmentPtr data_fragment = vector.at(i);
    jlong id[] = {data_fragment_holder_.Insert(data_fragment)};
    env->SetLongArrayRegion(ret, i, 1, id);
  }
  return ret;
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeFragment
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeFragment
    (JNIEnv *, jobject, jlong id) {
  data_fragment_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    getScanTasks
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_getScanTasks
    (JNIEnv* env, jobject, jlong fragment_id) {
  arrow::dataset::DataFragmentPtr data_fragment = data_fragment_holder_.Lookup(fragment_id);
  arrow::dataset::ScanTaskIterator itr = data_fragment->Scan(std::make_shared<arrow::dataset::ScanContext>())
      .ValueOrDie(); // fixme ValueOrDie
  std::vector<arrow::dataset::ScanTaskPtr> vector = collect(std::move(itr));
  jlongArray ret = env->NewLongArray(vector.size());
  for (unsigned long i = 0; i < vector.size(); i++) {
    arrow::dataset::ScanTaskPtr scan_task = vector.at(i);
    jlong id[] = {scan_task_holder_.Insert(scan_task)};
    env->SetLongArrayRegion(ret, i, 1, id);
  }
  return ret;
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeScanTask
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeScanTask
    (JNIEnv *, jobject, jlong id) {
  scan_task_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    scan
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_scan
    (JNIEnv *, jobject, jlong scan_task_id) {
  arrow::dataset::ScanTaskPtr scan_task = scan_task_holder_.Lookup(scan_task_id);
  arrow::RecordBatchIterator record_batch_iterator = scan_task->Scan().ValueOrDie(); // fixme ValueOrDie
  return iterator_holder_
      .Insert(std::make_shared<arrow::RecordBatchIterator>(std::move(record_batch_iterator))); // move and propagate
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeIterator
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeIterator
    (JNIEnv *, jobject, jlong id) {
  iterator_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    nextRecordBatch
 * Signature: (J)Lorg/apache/arrow/dataset/jni/NativeRecordBatchHandle;
 */
JNIEXPORT jobject JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_nextRecordBatch
    (JNIEnv* env, jobject, jlong iterator_id) {
  std::shared_ptr<arrow::RecordBatchIterator> itr = iterator_holder_.Lookup(iterator_id);
  std::shared_ptr<arrow::RecordBatch> record_batch;
  auto status = itr->Next(&record_batch);
  if (!status.ok()) {
    return nullptr; // fixme throw an error
  }
  if (record_batch == nullptr) {
    return nullptr; // stream ended
  }
  std::shared_ptr<arrow::Schema> schema = record_batch->schema();
  jobjectArray field_array =
      env->NewObjectArray(schema->num_fields(), record_batch_handle_field_class, nullptr);

  std::vector<std::shared_ptr<arrow::Buffer>> buffers;
  for (int i = 0; i < schema->num_fields(); ++i) {
    auto column = record_batch->column(i);
    auto dataArray = column->data();
    jobject field = env->NewObject(record_batch_handle_field_class, record_batch_handle_field_constructor,
                                   column->length(), column->null_count());
    env->SetObjectArrayElement(field_array, i, field);

    for (auto& buffer : dataArray->buffers) {
      buffers.push_back(buffer);
    }
  }

  jobjectArray buffer_array =
      env->NewObjectArray(buffers.size(), record_batch_handle_buffer_class, nullptr);

  for (size_t j = 0; j < buffers.size(); ++j) {
    auto buffer = buffers[j];
    jobject buffer_handle = env->NewObject(record_batch_handle_buffer_class, record_batch_handle_buffer_constructor,
                                           buffer_holder_.Insert(buffer), buffer->data(),
                                           buffer->size(), buffer->capacity());
    env->SetObjectArrayElement(buffer_array, j, buffer_handle);
  }

  jobject ret = env->NewObject(record_batch_handle_class, record_batch_handle_constructor,
                               record_batch->num_rows(), field_array, buffer_array);
  return ret;
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    releaseBuffer
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_releaseBuffer
    (JNIEnv *, jobject, jlong id) {
  buffer_holder_.Erase(id);
}

/*
 * Class:     org_apache_arrow_dataset_file_JniWrapper
 * Method:    makeFileSetDataSourceDiscovery
 * Signature: ([Ljava/lang/String;II)J
 */
JNIEXPORT jlong JNICALL Java_org_apache_arrow_dataset_file_JniWrapper_makeFileSetDataSourceDiscovery
    (JNIEnv* env, jobject, jstring path, jint file_format_id, jint file_system_id) {
  std::shared_ptr<arrow::dataset::FileFormat> file_format = GetFileFormat(env, file_format_id);
  std::string out_path;
  arrow::fs::FileSystemPtr fs = GetFileSystem(env, file_system_id, JStringToCString(env, path), &out_path);
  std::shared_ptr<arrow::dataset::DataSourceDiscovery>
      d = arrow::dataset::SingleFileDataSourceDiscovery::Make(out_path, fs, file_format).ValueOrDie();// fixme ValueOrDie
  return data_source_discovery_holder_.Insert(d);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    createScanner
 * Signature: ([J[B[Ljava/lang/String;[BJ)J
 */
JNIEXPORT jlong JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_createScanner
    (JNIEnv* env, jobject, jlongArray data_source_ids, jbyteArray schema_bytes, jobjectArray columns, jbyteArray filter, jlong batch_size) {
  std::shared_ptr<arrow::dataset::ScanContext> context = std::make_shared<arrow::dataset::ScanContext>();
  jsize size = env->GetArrayLength(data_source_ids);
  jboolean copied;
  jlong *ids_long = env->GetLongArrayElements(data_source_ids, &copied);

  std::vector<arrow::dataset::DataSourcePtr> data_source_vector;
  for (int i = 0; i < size; i++) {
    jlong id = ids_long[i];
    data_source_vector.push_back(data_source_holder_.Lookup(id));
  }
  if (copied) {
    env->ReleaseLongArrayElements(data_source_ids, ids_long, JNI_ABORT);
  }
  std::shared_ptr<arrow::Schema> schema;
  FromSchemaByteArray(env, schema_bytes, &schema).ok(); // fixme ok()
  arrow::dataset::DatasetPtr dataset = arrow::dataset::Dataset::Make(data_source_vector, schema).ValueOrDie(); // fixme ValueOrDie()
  std::shared_ptr<arrow::dataset::ScannerBuilder> scanner_builder = dataset->NewScan().ValueOrDie(); // fixme ValueOrDie()

  std::vector<std::string> column_vector = ToStringVector(env, columns);
  scanner_builder->Project(column_vector).ok(); // fixme ok()
  scanner_builder->BatchSize(batch_size).ok(); // fixme ok()
  // todo initialize filters
  auto scanner = scanner_builder->Finish().ValueOrDie();
  return scanner_holder_.Insert(scanner);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    closeScanner
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_closeScanner
    (JNIEnv *, jobject, jlong scanner_id) {
  scanner_holder_.Erase(scanner_id);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    getSchemaFromScanner
 * Signature: (J)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_getSchemaFromScanner
    (JNIEnv* env, jobject, jlong scanner_id) {
  std::shared_ptr<arrow::Schema> schema = scanner_holder_.Lookup(scanner_id)->GetSchema().ValueOrDie(); // fixme ValueOrDie()
  return ToSchemaByteArray(env, schema);
}

/*
 * Class:     org_apache_arrow_dataset_jni_JniWrapper
 * Method:    getScanTasksFromScanner
 * Signature: (J)[J
 */
JNIEXPORT jlongArray JNICALL Java_org_apache_arrow_dataset_jni_JniWrapper_getScanTasksFromScanner
    (JNIEnv* env, jobject, jlong scanner_id) {
  auto scanner = scanner_holder_.Lookup(scanner_id);
  arrow::dataset::ScanTaskIterator itr = scanner->Scan().ValueOrDie(); // fixme ValueOrDie
  std::vector<arrow::dataset::ScanTaskPtr> vector = collect(std::move(itr));
  // Duplicated code with :323-:330
  jlongArray ret = env->NewLongArray(vector.size());
  for (unsigned long i = 0; i < vector.size(); i++) {
    arrow::dataset::ScanTaskPtr scan_task = vector.at(i);
    jlong id[] = {scan_task_holder_.Insert(scan_task)};
    env->SetLongArrayRegion(ret, i, 1, id);
  }
  return ret;
}