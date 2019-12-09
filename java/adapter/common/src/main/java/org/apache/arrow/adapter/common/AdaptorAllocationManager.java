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

package org.apache.arrow.adapter.common;

import org.apache.arrow.memory.AllocationManager;
import org.apache.arrow.memory.BaseAllocator;
import org.apache.arrow.memory.BufferLedger;

/**
 * A simple allocation manager implementation for memory allocated by native code.
 * The underlying memory will be released when reference count reach zero.
 */
public class AdaptorAllocationManager extends AllocationManager {

  private final long nativeMemoryHolder;
  private final long memoryAddress;

  protected AdaptorAllocationManager(long nativeMemoryHolder,
                                     BaseAllocator accountingAllocator,
                                     long memoryAddress,
                                     int size) {
    super(accountingAllocator, size);
    this.nativeMemoryHolder = nativeMemoryHolder;
    this.memoryAddress = memoryAddress;
  }

  @Override
  protected long memoryAddress() {
    return memoryAddress;
  }

  @Override
  protected void release0() {
    nativeRelease(nativeMemoryHolder);
  }

  @Override
  public BufferLedger associate(BaseAllocator allocator, boolean retain) {
    return super.associate(allocator, retain);
  }

  private native void nativeRelease(long nativeMemoryHolder);
}
