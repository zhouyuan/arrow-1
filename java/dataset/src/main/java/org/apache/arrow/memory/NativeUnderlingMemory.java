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

package org.apache.arrow.memory;

import io.netty.buffer.UnsafeDirectLittleEndian;
import org.apache.arrow.dataset.jni.JniWrapper;

public class NativeUnderlingMemory extends AllocationManagerBase {

  private final int size;
  private final long nativeInstanceId;

  public NativeUnderlingMemory(BaseAllocator accountingAllocator, int size, long nativeInstanceId) {
    super(accountingAllocator);
    this.size = size;
    this.nativeInstanceId = nativeInstanceId;
  }

  @Override
  public BufferLedger associate(BaseAllocator allocator) {
    return super.associate(allocator);
  }

  @Override
  void release0() {
    JniWrapper.get().releaseBuffer(nativeInstanceId);
  }

  @Override
  UnsafeDirectLittleEndian getMemoryChunk() {
    return AllocationManager.EMPTY; // not supported
  }

  @Override
  int getSize() {
    return size;
  }
}
