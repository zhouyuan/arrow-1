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

import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.apache.arrow.dataset.datasource.DataSource;
import org.apache.arrow.dataset.fragment.DataFragment;
import org.apache.arrow.dataset.scanner.ScanOptions;

/**
 * Native implementation of {@link DataSource}.
 */
public class NativeDataSource implements DataSource, AutoCloseable {

  private final NativeContext context;
  private final long dataSourceId;

  public NativeDataSource(NativeContext context, long dataSourceId) {
    this.context = context;
    this.dataSourceId = dataSourceId;
  }

  @Override
  public Iterable<? extends DataFragment> getFragments(ScanOptions options) {
    return LongStream.of(JniWrapper.get()
      .getFragments(dataSourceId, options.getBatchSize()))
      .mapToObj(id -> new NativeDataFragment(context, id, options.getColumns(), options.getFilter()))
      .collect(Collectors.toList());
  }

  public long getDataSourceId() {
    return dataSourceId;
  }

  @Override
  public void close() throws Exception {
    JniWrapper.get().closeDataSource(dataSourceId);
  }
}
