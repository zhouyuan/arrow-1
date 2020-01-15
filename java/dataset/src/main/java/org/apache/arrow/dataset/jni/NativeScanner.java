/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.arrow.dataset.jni;

import java.io.IOException;
import java.util.Arrays;
import java.util.stream.Collectors;

import org.apache.arrow.dataset.Dataset;
import org.apache.arrow.dataset.scanner.ScanOptions;
import org.apache.arrow.dataset.scanner.ScanTask;
import org.apache.arrow.dataset.scanner.Scanner;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.util.SchemaUtils;
import org.apache.arrow.vector.types.pojo.Schema;

/**
 * Native implementation of {@link Scanner}.
 */
public class NativeScanner implements Scanner {
  private final long nativeInstanceId;
  private final BufferAllocator allocator;

  public NativeScanner(Dataset<NativeDataSource> dataset, ScanOptions options, BufferAllocator allocator) {
    this.allocator = allocator;
    this.nativeInstanceId = init(dataset, options);
  }

  private long init(Dataset<NativeDataSource> dataset, ScanOptions options) {
    final long[] dataSourceIds = dataset.getSources().stream().mapToLong(NativeDataSource::getDataSourceId).toArray();
    try {
      byte[] schema = SchemaUtils.get().serialize(dataset.getSchema());
      return JniWrapper.get().createScanner(dataSourceIds, schema, options.getColumns(),
          options.getFilter().toByteArray(), options.getBatchSize());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public Iterable<? extends ScanTask> scan() {
    final byte[] schemaBytes = JniWrapper.get().getSchemaFromScanner(nativeInstanceId);
    try {
      Schema schema = SchemaUtils.get().deserialize(schemaBytes, allocator);
      return Arrays.stream(JniWrapper.get().getScanTasksFromScanner(nativeInstanceId))
          .mapToObj(id -> new NativeScanTask(new NativeContext(schema, allocator), id))
          .collect(Collectors.toList());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void close() throws Exception {
    JniWrapper.get().closeScanner(nativeInstanceId);
  }
}
