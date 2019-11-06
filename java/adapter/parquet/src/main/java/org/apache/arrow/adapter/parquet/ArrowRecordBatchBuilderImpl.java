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

package org.apache.arrow.adapter.parquet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.BufferManager;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.ipc.message.ArrowFieldNode;
import org.apache.arrow.vector.ipc.message.ArrowRecordBatch;

import io.netty.buffer.ArrowBuf;

/**
 * ArrowRecordBatchBuilderImpl used to wrap native returned data into an ArrowRecordBatch.
 */
public class ArrowRecordBatchBuilderImpl {

  private static final BufferAllocator EMPTY_ALLOCATOR = new RootAllocator() {
    @Override
    public ArrowBuf buffer(int initialRequestSize, BufferManager manager) {
      throw new UnsupportedOperationException();
    }
  };

  private int length;
  private ArrowRecordBatchBuilder recordBatchBuilder;

  /**
   * Create ArrowRecordBatchBuilderImpl instance from ArrowRecordBatchBuilder.
   * @param recordBatchBuilder ArrowRecordBatchBuilder instance.
   */
  public ArrowRecordBatchBuilderImpl(ArrowRecordBatchBuilder recordBatchBuilder) {
    this.recordBatchBuilder = recordBatchBuilder;
  }

  /**
   * Build ArrowRecordBatch from ArrowRecordBatchBuilder instance.
   * @throws IOException throws exception
   */
  public ArrowRecordBatch build() throws IOException {
    if (recordBatchBuilder.length == 0) {
      return null;
    }

    List<ArrowFieldNode> nodes = new ArrayList<ArrowFieldNode>();
    for (ArrowFieldNodeBuilder tmp : recordBatchBuilder.nodeBuilders) {
      nodes.add(new ArrowFieldNode(tmp.length, tmp.nullCount));
    }

    List<ArrowBuf> buffers = new ArrayList<ArrowBuf>();
    for (ArrowBufBuilder tmp : recordBatchBuilder.bufferBuilders) {
      if (tmp == null) {
        // for most primitive types, null indicates an empty validity buffer
        buffers.add(EMPTY_ALLOCATOR.getEmpty());
      } else {
        AdaptorReferenceManager referenceManager =
            new AdaptorReferenceManager(tmp.nativeInstanceId, tmp.size);
        buffers.add(new ArrowBuf(referenceManager, null, tmp.size, tmp.memoryAddress, false));
      }
    }
    return new ArrowRecordBatch(recordBatchBuilder.length, nodes, buffers);
  }
} 
