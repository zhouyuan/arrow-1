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

import org.apache.arrow.dataset.filter.Filter;
import org.apache.arrow.dataset.fragment.DataFragment;
import org.apache.arrow.dataset.scanner.ProjectAndFilterScanTask;
import org.apache.arrow.dataset.scanner.ScanTask;

/**
 * Native implementation of {@link DataFragment}.
 */
public class NativeDataFragment implements DataFragment, AutoCloseable {
  private final NativeContext context;
  private final long fragmentId;
  private final String[] columns;
  private final Filter filter;

  /**
   * Constructor.
   *
   * @param context Native context
   * @param fragmentId Native ID of the fragment
   * @param columns Projected columns
   * @param filter Filter
   */
  public NativeDataFragment(NativeContext context, long fragmentId, String[] columns, Filter filter) {
    this.context = context;
    this.fragmentId = fragmentId;
    this.columns = columns;
    this.filter = filter;
  }

  @Override
  public Iterable<? extends ScanTask> scan() {
    return LongStream.of(JniWrapper.get().getScanTasks(fragmentId))
        .mapToObj(id -> new ProjectAndFilterScanTask(new NativeScanTask(context, id), columns, filter))
        .collect(Collectors.toList());
  }

  @Override
  public void close() throws Exception {
    JniWrapper.get().closeFragment(fragmentId);
  }
}
