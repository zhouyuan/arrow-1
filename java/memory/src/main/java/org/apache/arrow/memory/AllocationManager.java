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

import io.netty.buffer.PooledByteBufAllocatorL;
import io.netty.buffer.UnsafeDirectLittleEndian;

/**
 * Manages the relationship between one or more allocators and a particular UDLE. Ensures that
 * one allocator owns the
 * memory that multiple allocators may be referencing. Manages a BufferLedger between each of its
 * associated allocators.
 * This class is also responsible for managing when memory is allocated and returned to the
 * Netty-based
 * PooledByteBufAllocatorL.
 *
 * <p>The only reason that this isn't package private is we're forced to put ArrowBuf in Netty's
 * package which need access
 * to these objects or methods.
 *
 * <p>Threading: AllocationManager manages thread-safety internally. Operations within the context
 * of a single BufferLedger
 * are lockless in nature and can be leveraged by multiple threads. Operations that cross the
 * context of two ledgers
 * will acquire a lock on the AllocationManager instance. Important note, there is one
 * AllocationManager per
 * UnsafeDirectLittleEndian buffer allocation. As such, there will be thousands of these in a
 * typical query. The
 * contention of acquiring a lock on AllocationManager should be very low.
 */
public class AllocationManager extends AllocationManagerBase {

  private static final PooledByteBufAllocatorL INNER_ALLOCATOR = new PooledByteBufAllocatorL();

  static final UnsafeDirectLittleEndian EMPTY = INNER_ALLOCATOR.empty;
  static final long CHUNK_SIZE = INNER_ALLOCATOR.getChunkSize();

  private final int size;
  private final UnsafeDirectLittleEndian memoryChunk;

  AllocationManager(BaseAllocator accountingAllocator, int size) {
    super(accountingAllocator);
    accountingAllocator.assertOpen();
    this.memoryChunk = INNER_ALLOCATOR.allocate(size);
    this.size = memoryChunk.capacity();
  }

  @Override
  void release0() {
    memoryChunk.release();
  }

  /**
   * Get the underlying memory chunk managed by this AllocationManager.
   * @return buffer
   */
  @Override
  UnsafeDirectLittleEndian getMemoryChunk() {
    return memoryChunk;
  }

  /**
   * Return the size of underlying chunk of memory managed by this Allocation Manager.
   * @return size of memory chunk
   */
  @Override
  public int getSize() {
    return size;
  }
}
