//go:build darwin && cgo
// +build darwin,cgo

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CoreFoundation/CoreFoundation.h>
#include "metal.h"
#include <stdio.h>
#include <string.h>

// ─── Global state ───────────────────────────────────────────────────────────

static id<MTLDevice>       g_device       = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLLibrary>      g_library      = nil;
static id<MTLBuffer>       g_dummyParamBuffer = nil;

// Pipeline cache: we lazily create pipelines on first use.
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipelines = nil;
static NSLock* g_pipelineLock = nil;

// Optional batched command buffer for one top-level Execute (many kernels, one wait).
static id<MTLCommandBuffer> g_encode_cb = nil;

// ─── Device management ──────────────────────────────────────────────────────

int metal_device_count(void) {
    NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
    if (!devices) return 0;
    return (int)devices.count;
}

int metal_init(const char* metallib_path) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) return -1;

        g_commandQueue = [g_device newCommandQueue];
        if (!g_commandQueue) { g_device = nil; return -2; }

        g_pipelineLock = [[NSLock alloc] init];
        g_pipelines = [[NSMutableDictionary alloc] init];

        NSString* path = [NSString stringWithUTF8String:metallib_path];
        NSError* error = nil;
        g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!g_library) {
            NSLog(@"gomlx-metal: failed to load metallib: %@", error);
            g_pipelines = nil;
            g_pipelineLock = nil;
            g_commandQueue = nil;
            g_device = nil;
            return -3;
        }

        g_dummyParamBuffer = [g_device newBufferWithLength:sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        if (!g_dummyParamBuffer) {
            g_library = nil;
            g_pipelines = nil;
            g_pipelineLock = nil;
            g_commandQueue = nil;
            g_device = nil;
            return -4;
        }
        memset([g_dummyParamBuffer contents], 0, sizeof(float));

        return 0;
    }
}

const char* metal_device_name(void) {
    if (!g_device) return strdup("unavailable");
    return strdup([[g_device name] UTF8String]);
}

void metal_finalize(void) {
    @autoreleasepool {
        if (g_encode_cb) {
            [g_encode_cb commit];
            [g_encode_cb waitUntilCompleted];
            g_encode_cb = nil;
        }
        [g_pipelines removeAllObjects];
        g_pipelines = nil;
        g_pipelineLock = nil;
        g_dummyParamBuffer = nil;
        g_library = nil;
        g_commandQueue = nil;
        g_device = nil;
    }
}

// ─── Pipeline cache ─────────────────────────────────────────────────────────

static id<MTLComputePipelineState> get_pipeline(NSString* name) {
    if (!name) return nil;
    if (!g_pipelineLock || !g_pipelines) {
        NSLog(@"gomlx-metal: pipeline cache unavailable for %@", name);
        return nil;
    }
    [g_pipelineLock lock];
    id<MTLComputePipelineState> pipeline = g_pipelines[name];
    [g_pipelineLock unlock];

    if (pipeline) return pipeline;
    if (!g_library || !g_device) {
        NSLog(@"gomlx-metal: pipeline lookup after shutdown for %@", name);
        return nil;
    }

    id<MTLFunction> fn = [g_library newFunctionWithName:name];
    if (!fn) {
        NSLog(@"gomlx-metal: kernel not found: %@", name);
        return nil;
    }

    NSError* error = nil;
    pipeline = [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        NSLog(@"gomlx-metal: pipeline creation failed for %@: %@", name, error);
        return nil;
    }

    [g_pipelineLock lock];
    id<MTLComputePipelineState> existing = g_pipelines[name];
    if (existing) {
        [g_pipelineLock unlock];
        return existing;
    }
    g_pipelines[name] = pipeline;
    [g_pipelineLock unlock];

    return pipeline;
}

// ─── Dispatch helpers ───────────────────────────────────────────────────────

static int dispatch_1d(id<MTLComputePipelineState> pipeline,
                       id<MTLComputeCommandEncoder> enc,
                       NSUInteger count) {
    [enc setComputePipelineState:pipeline];
    NSUInteger tg = pipeline.threadExecutionWidth;
    if (tg > pipeline.maxTotalThreadsPerThreadgroup)
        tg = pipeline.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = count;
    if (tg == 0) tg = 1;

    [enc dispatchThreads:MTLSizeMake(count, 1, 1)
   threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    return 0;
}

static int dispatch_threadgroups(id<MTLComputePipelineState> pipeline,
                                 id<MTLComputeCommandEncoder> enc,
                                 NSUInteger num_groups, NSUInteger group_size) {
    [enc setComputePipelineState:pipeline];
    [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group_size, 1, 1)];
    return 0;
}

static int commit_and_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return (cb.status == MTLCommandBufferStatusCompleted) ? 0 : -5;
}

void metal_encode_begin(void) {
    if (!g_commandQueue) return;
    if (g_encode_cb != nil) {
        NSLog(@"gomlx-metal: nested metal_encode_begin (ignored)");
        return;
    }
    g_encode_cb = [g_commandQueue commandBuffer];
}

int metal_encode_barrier_wait(void) {
    if (!g_encode_cb) return 0;
    int r = commit_and_wait(g_encode_cb);
    if (r != 0) {
        g_encode_cb = nil;
        return r;
    }
    g_encode_cb = [g_commandQueue commandBuffer];
    return 0;
}

int metal_encode_end_wait(void) {
    if (!g_encode_cb) return 0;
    int r = commit_and_wait(g_encode_cb);
    g_encode_cb = nil;
    return r;
}

// ─── Buffer management ──────────────────────────────────────────────────────

MetalBuffer metal_buffer_alloc(size_t bytes) {
    if (!g_device || bytes == 0) return NULL;
    id<MTLBuffer> buf = [g_device newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!buf) return NULL;
    return (MetalBuffer)CFBridgingRetain(buf);
}

void metal_buffer_free(MetalBuffer buf) {
    if (!buf) return;
    CFBridgingRelease(buf);
}

void* metal_buffer_contents(MetalBuffer buf) {
    if (!buf) return NULL;
    id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buf;
    return [mtlBuf contents];
}

size_t metal_buffer_length(MetalBuffer buf) {
    if (!buf) return 0;
    id<MTLBuffer> mtlBuf = (__bridge id<MTLBuffer>)buf;
    return [mtlBuf length];
}

// ─── Dtype suffix helper ────────────────────────────────────────────────────

static NSString* dtype_suffix(int dtype) {
    switch (dtype) {
        case 0: return @"_f16";
        case 1: return @"_f32";
        case 2: return @"_f64";
        case 3: return @"_u32";
        case 4: return @""; // bool / uchar elementwise (kernels without _f32/_u32 suffix)
        case 5: return @"_i32";
        default: return @"";
    }
}

// op_where_pred_* kernel suffix; codes must match wherePredValueKind in metal.go.
static NSString* where_pred_value_suffix(int vk) {
    switch (vk) {
        case 0: return @"_f16";
        case 1: return @"_f32";
        case 2: return @"_i32";
        case 3: return @"_i64";
        case 4: return @"_bool";
        case 5: return @"_i8";
        case 6: return @"_i16";
        case 7: return @"_u8";
        case 8: return @"_u16";
        case 9: return @"_u32";
        case 10: return @"_u64";
        default: return @"";
    }
}

// ─── Elementwise unary ops ──────────────────────────────────────────────────

int metal_unary_op(const char* op_name, MetalBuffer src, MetalBuffer dst,
                   uint32_t num_elements, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"%s%@",
                                 op_name, dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Elementwise binary ops ─────────────────────────────────────────────────

int metal_binary_op(const char* op_name, MetalBuffer lhs, MetalBuffer rhs,
                    MetalBuffer dst, uint32_t num_elements, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"%s%@",
                                 op_name, dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)lhs offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)rhs offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:2];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Reductions ─────────────────────────────────────────────────────────────

int metal_reduce_op(const char* op_name, MetalBuffer src, MetalBuffer dst,
                    uint32_t outer_size, uint32_t inner_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"%s%@",
                                 op_name, dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&outer_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&inner_size length:sizeof(uint32_t) atIndex:3];

        // The reduction kernels halve the active lane count each round, so the
        // threadgroup width must be a power of two to avoid dropping lanes.
        NSUInteger group_size = 1;
        while ((group_size << 1) <= 256 && (group_size << 1) <= inner_size) {
            group_size <<= 1;
        }
        dispatch_threadgroups(pipeline, enc, outer_size, group_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── DotGeneral ─────────────────────────────────────────────────────────────

int metal_dot_general(MetalBuffer a, MetalBuffer b, MetalBuffer c,
                      uint32_t batch, uint32_t m, uint32_t k, uint32_t n,
                      int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"dot_general%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)b offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)c offset:0 atIndex:2];
        [enc setBytes:&batch length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&m     length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&k     length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&n     length:sizeof(uint32_t) atIndex:6];

        [enc setComputePipelineState:pipeline];
        uint32_t tile = 16;
        MTLSize gridSize = MTLSizeMake(
            ((n + tile - 1) / tile) * tile,
            ((m + tile - 1) / tile) * tile,
            batch);
        MTLSize tgSize = MTLSizeMake(tile, tile, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused Softmax ──────────────────────────────────────────────────────────

int metal_fused_softmax(MetalBuffer src, MetalBuffer dst,
                        uint32_t outer_size, uint32_t axis_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"fused_softmax%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&outer_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&axis_size  length:sizeof(uint32_t) atIndex:3];

        NSUInteger group_size = 256;
        if (group_size > axis_size) group_size = axis_size;
        dispatch_threadgroups(pipeline, enc, outer_size, group_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused LayerNorm ────────────────────────────────────────────────────────

int metal_fused_layernorm(MetalBuffer x, MetalBuffer gamma, MetalBuffer beta,
                          MetalBuffer dst, uint32_t batch_size, uint32_t norm_size,
                          float epsilon, int has_gamma, int has_beta, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"fused_layernorm%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        id<MTLBuffer> dummy = g_dummyParamBuffer;
        if (!dummy) return -1;

        [enc setBuffer:(__bridge id<MTLBuffer>)x     offset:0 atIndex:0];
        // gamma/beta may be NULL — set a dummy buffer if so
        if (has_gamma && gamma) {
            [enc setBuffer:(__bridge id<MTLBuffer>)gamma offset:0 atIndex:1];
        } else {
            [enc setBuffer:dummy offset:0 atIndex:1];
        }
        if (has_beta && beta) {
            [enc setBuffer:(__bridge id<MTLBuffer>)beta offset:0 atIndex:2];
        } else {
            [enc setBuffer:dummy offset:0 atIndex:2];
        }
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:3];
        [enc setBytes:&batch_size length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&norm_size  length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&epsilon    length:sizeof(float)    atIndex:6];
        [enc setBytes:&has_gamma  length:sizeof(int)      atIndex:7];
        [enc setBytes:&has_beta   length:sizeof(int)      atIndex:8];

        NSUInteger group_size = 256;
        if (group_size > norm_size) group_size = norm_size;
        dispatch_threadgroups(pipeline, enc, batch_size, group_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused RMSNorm ──────────────────────────────────────────────────────────

int metal_fused_rmsnorm(MetalBuffer x, MetalBuffer weight, MetalBuffer dst,
                        uint32_t batch_size, uint32_t norm_size,
                        float epsilon, int has_weight, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"fused_rmsnorm%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        if (has_weight && weight) {
            [enc setBuffer:(__bridge id<MTLBuffer>)weight offset:0 atIndex:1];
        } else {
            [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:1];
        }
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:2];
        [enc setBytes:&batch_size  length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&norm_size   length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&epsilon     length:sizeof(float)    atIndex:5];
        [enc setBytes:&has_weight  length:sizeof(int)      atIndex:6];

        NSUInteger group_size = 256;
        if (group_size > norm_size) group_size = norm_size;
        dispatch_threadgroups(pipeline, enc, batch_size, group_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused GELU ─────────────────────────────────────────────────────────────

int metal_fused_gelu(MetalBuffer src, MetalBuffer dst,
                     uint32_t num_elements, int exact, int dtype) {
    @autoreleasepool {
        NSString* variant = exact ? @"fused_gelu_exact" : @"fused_gelu_approx";
        NSString* kernel_name = [NSString stringWithFormat:@"%@%@",
                                 variant, dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused RoPE ─────────────────────────────────────────────────────────────

int metal_fused_rope(MetalBuffer x, MetalBuffer dst,
                     uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                     uint32_t head_dim, uint32_t rot_dim, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"fused_rope%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)x   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&batch     length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&seq_len   length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&num_heads length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&head_dim  length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&rot_dim   length:sizeof(uint32_t) atIndex:6];

        uint32_t total = batch * seq_len * num_heads * head_dim;
        dispatch_1d(pipeline, enc, total);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Fused SDPA ─────────────────────────────────────────────────────────────

int metal_transpose_perm(MetalBuffer src, MetalBuffer dst, MetalBuffer config,
                         uint32_t total_elements, uint32_t config_size, uint32_t elem_size) {
    (void)config_size;
    @autoreleasepool {
        NSString* kernel_name = @"transpose_perm_bytes";
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int gomlx_metal_transpose_perm(MetalBuffer src, MetalBuffer dst, MetalBuffer config,
                               uint32_t total_elements, uint32_t config_size, uint32_t elem_size) {
    return metal_transpose_perm(src, dst, config, total_elements, config_size, elem_size);
}

int metal_fused_sdpa(MetalBuffer q, MetalBuffer k, MetalBuffer v,
                     MetalBuffer mask_buf, MetalBuffer dst,
                     uint32_t batch, uint32_t num_heads, uint32_t num_kv_heads,
                     uint32_t seq_len, uint32_t kv_len, uint32_t head_dim,
                     float scale, int causal, int mask_type,
                     uint32_t mask_batch_stride, uint32_t mask_head_stride, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"fused_sdpa%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)q   offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)k   offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)v   offset:0 atIndex:2];
        if (mask_buf) {
            [enc setBuffer:(__bridge id<MTLBuffer>)mask_buf offset:0 atIndex:3];
        } else {
            [enc setBuffer:(__bridge id<MTLBuffer>)q offset:0 atIndex:3]; // dummy
        }
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:4];
        [enc setBytes:&batch        length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&num_heads    length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&num_kv_heads length:sizeof(uint32_t) atIndex:7];
        [enc setBytes:&seq_len      length:sizeof(uint32_t) atIndex:8];
        [enc setBytes:&kv_len       length:sizeof(uint32_t) atIndex:9];
        [enc setBytes:&head_dim     length:sizeof(uint32_t) atIndex:10];
        [enc setBytes:&scale        length:sizeof(float)    atIndex:11];
        [enc setBytes:&causal       length:sizeof(int)      atIndex:12];
        [enc setBytes:&mask_type    length:sizeof(int)      atIndex:13];
        [enc setBytes:&mask_batch_stride length:sizeof(uint32_t) atIndex:14];
        [enc setBytes:&mask_head_stride  length:sizeof(uint32_t) atIndex:15];

        uint32_t num_groups = batch * num_heads * seq_len;
        NSUInteger group_size = 256;
        dispatch_threadgroups(pipeline, enc, num_groups, group_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int gomlx_metal_fused_sdpa(MetalBuffer q, MetalBuffer k, MetalBuffer v,
                           MetalBuffer mask, MetalBuffer dst,
                           uint32_t batch, uint32_t num_heads, uint32_t num_kv_heads,
                           uint32_t seq_len, uint32_t kv_len, uint32_t head_dim,
                           float scale, int causal, int mask_type,
                           uint32_t mask_batch_stride, uint32_t mask_head_stride, int dtype) {
    return metal_fused_sdpa(q, k, v, mask, dst, batch, num_heads, num_kv_heads,
                            seq_len, kv_len, head_dim, scale, causal, mask_type,
                            mask_batch_stride, mask_head_stride, dtype);
}

// ─── FusedQuantizedDense (GPU) ─────────────────────────────────────────────

static NSString* metal_qdense_kernel_name(int kind) {
    switch (kind) {
        case 0: return @"q_dense_nf4_u8_f32";
        case 1: return @"q_dense_nf4_pack_f32";
        case 2: return @"q_dense_lin_i8_f32";
        case 3: return @"q_dense_lin_u8_f32";
        case 4: return @"q_dense_lin_i4p_f32";
        case 5: return @"q_dense_lin_u4p_f32";
        default: return nil;
    }
}

int metal_quantized_dense(MetalBuffer x, MetalBuffer w, MetalBuffer scales,
                          MetalBuffer zp, MetalBuffer bias, MetalBuffer out,
                          const uint32_t* config, uint32_t config_num_uints,
                          uint32_t grid_mn, int kind) {
    @autoreleasepool {
        NSString* kn = metal_qdense_kernel_name(kind);
        if (!kn) return -1;
        id<MTLComputePipelineState> pipeline = get_pipeline(kn);
        if (!pipeline) return -1;

        id<MTLBuffer> xBuf = (__bridge id<MTLBuffer>)x;
        if (!zp) zp = x;
        if (!bias) bias = x;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:xBuf offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)w offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)scales offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)zp offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:5];
        [enc setBytes:config length:sizeof(uint32_t) * config_num_uints atIndex:6];

        dispatch_1d(pipeline, enc, grid_mn);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Broadcast ──────────────────────────────────────────────────────────────

int metal_broadcast(MetalBuffer src, MetalBuffer dst,
                    uint32_t src_size, uint32_t dst_size, uint32_t elem_size) {
    @autoreleasepool {
        NSString* kernel_name = @"broadcast_bytes";
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&src_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:3];

        dispatch_1d(pipeline, enc, dst_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Iota ───────────────────────────────────────────────────────────────────

static NSString* iota_dtype_suffix(int dtype) {
    switch (dtype) {
        case 0: return @"_f16";
        case 1: return @"_f32";
        case 3: return @"_i32";
        case 4: return @"_i64";
        case 5: return @"_u32";
        case 6: return @"_u64";
        default: return @"__invalid_iota__";
    }
}

int metal_iota(MetalBuffer dst, uint32_t batch_size, uint32_t iota_size,
               uint32_t repeat_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"iota%@",
                                 iota_dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:0];
        [enc setBytes:&batch_size  length:sizeof(uint32_t) atIndex:1];
        [enc setBytes:&iota_size   length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&repeat_size length:sizeof(uint32_t) atIndex:3];

        uint32_t total = batch_size * iota_size * repeat_size;
        dispatch_1d(pipeline, enc, total);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── ConvertDType ───────────────────────────────────────────────────────────

static NSString* convert_kernel_name(int src_dtype, int dst_dtype) {
    // src/dst kind codes must match dtypeToMetalExt in executable.go.
    // Float64/double is intentionally unsupported here.
    if (src_dtype >= 0 && src_dtype <= 11 &&
        dst_dtype >= 0 && dst_dtype <= 11 &&
        src_dtype != 2 && dst_dtype != 2) {
        return @"convert_dtype";
    }
    return nil;
}

int metal_convert_dtype(MetalBuffer src, MetalBuffer dst,
                        uint32_t num_elements, int src_dtype, int dst_dtype) {
    @autoreleasepool {
        NSString* kernel_name = convert_kernel_name(src_dtype, dst_dtype);
        if (!kernel_name) return -1;

        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&src_dtype length:sizeof(int) atIndex:2];
        [enc setBytes:&dst_dtype length:sizeof(int) atIndex:3];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── ArgMinMax ──────────────────────────────────────────────────────────────

int metal_argminmax(MetalBuffer src, MetalBuffer dst,
                    uint32_t prefix_size, uint32_t reduce_size,
                    uint32_t suffix_size, int is_min, int dtype) {
    @autoreleasepool {
        NSString* op = is_min ? @"argmin" : @"argmax";
        NSString* kernel_name = [NSString stringWithFormat:@"%@%@",
                                 op, dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&prefix_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&reduce_size length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&suffix_size length:sizeof(uint32_t) atIndex:4];

        uint32_t total = prefix_size * suffix_size;
        dispatch_1d(pipeline, enc, total);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Reverse inner axis ─────────────────────────────────────────────────────

int metal_reverse_inner(MetalBuffer src, MetalBuffer dst,
                        uint32_t outer_size, uint32_t inner_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"reverse_inner%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];
        [enc setBytes:&outer_size length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&inner_size length:sizeof(uint32_t) atIndex:3];

        uint32_t total = outer_size * inner_size;
        dispatch_1d(pipeline, enc, total);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── BroadcastInDim ─────────────────────────────────────────────────────────

int metal_broadcast_in_dim(MetalBuffer src, MetalBuffer dst,
                           MetalBuffer config, uint32_t total_elements,
                           uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        NSString* kernel_name = @"broadcast_in_dim_bytes";
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Concatenate ────────────────────────────────────────────────────────────

int metal_concatenate(MetalBuffer src, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        NSString* kernel_name = @"concatenate_bytes";
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Slice ──────────────────────────────────────────────────────────────────

int metal_slice(MetalBuffer src, MetalBuffer dst,
                MetalBuffer config, uint32_t total_elements,
                uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"slice_bytes");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Pad ────────────────────────────────────────────────────────────────────

int metal_pad(MetalBuffer src, MetalBuffer pad_value, MetalBuffer dst,
              MetalBuffer config, uint32_t total_elements,
              uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"pad_bytes");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src       offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst       offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config    offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)pad_value offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:5];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Reverse ────────────────────────────────────────────────────────────────

int metal_reverse(MetalBuffer src, MetalBuffer dst,
                  MetalBuffer config, uint32_t total_elements,
                  uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"reverse_bytes");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Where (ternary select) ─────────────────────────────────────────────────

int metal_where(MetalBuffer pred, MetalBuffer on_true, MetalBuffer on_false,
                MetalBuffer dst, uint32_t num_elements, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"op_where%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)pred     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)on_true  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)on_false offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst      offset:0 atIndex:3];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_where_bool_pred(MetalBuffer pred, MetalBuffer on_true, MetalBuffer on_false,
                          MetalBuffer dst, uint32_t num_elements, int value_kind) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"op_where_pred%@",
                                 where_pred_value_suffix(value_kind)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)pred     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)on_true  offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)on_false offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst      offset:0 atIndex:3];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_cast_i32_to_i64(MetalBuffer src, MetalBuffer dst, uint32_t num_elements) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"cast_i32_to_i64");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_bool_mask_to_float(MetalBuffer src, MetalBuffer dst, uint32_t num_elements, int dtype) {
    @autoreleasepool {
        NSString* kn = (dtype == 0) ? @"op_bool_to_f16" : @"op_bool_to_f32";
        id<MTLComputePipelineState> pipeline = get_pipeline(kn);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst offset:0 atIndex:1];

        dispatch_1d(pipeline, enc, num_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_gather_axis_row_bytes(MetalBuffer src, MetalBuffer row, uint32_t base_off,
                                uint32_t axis_stride, uint32_t axis_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"gather_axis_row_bytes");
        if (!pipeline) return -1;

        struct { uint32_t x, y, z, w; } p = { base_off, axis_stride, axis_size, elem_size };

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)row offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];

        dispatch_1d(pipeline, enc, axis_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_scatter_axis_row_perm_bytes(MetalBuffer row, MetalBuffer dst, MetalBuffer indices,
                                      uint32_t base_off, uint32_t axis_stride,
                                      uint32_t axis_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"scatter_axis_row_perm_bytes");
        if (!pipeline) return -1;

        struct { uint32_t x, y, z, w; } p = { base_off, axis_stride, axis_size, elem_size };

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)row     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst     offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];

        dispatch_1d(pipeline, enc, axis_size);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_sort_load_pair_bytes(MetalBuffer flat, MetalBuffer lhs, MetalBuffer rhs, MetalBuffer idx,
                               uint32_t base_elem, uint32_t axis_stride, uint32_t elem_size,
                               uint32_t sort_i, uint32_t sort_j) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"sort_load_pair_bytes");
        if (!pipeline) return -1;

        struct {
            uint32_t x, y, z, w;
        } p4 = { base_elem, axis_stride, elem_size, 0 };
        struct {
            uint32_t x, y;
        } ij = { sort_i, sort_j };

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)flat offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)lhs offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)rhs offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)idx offset:0 atIndex:3];
        [enc setBytes:&p4 length:sizeof(p4) atIndex:4];
        [enc setBytes:&ij length:sizeof(ij) atIndex:5];

        dispatch_1d(pipeline, enc, 1);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// Odd-even transposition: if swap_when_pred_nonzero, swap when pred (strict less(lhs,rhs));
// else swap when !pred (unstable: lhs>=rhs in comparator convention).
int metal_sort_adjacent_swap_idx(MetalBuffer idx, MetalBuffer pred, uint32_t pair_i, uint32_t n,
                                  uint32_t swap_when_pred_nonzero) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"sort_adjacent_swap_idx");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)idx offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)pred offset:0 atIndex:1];
        [enc setBytes:&pair_i length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&swap_when_pred_nonzero length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, 1);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_sort_bitonic_swap_idx(MetalBuffer idx, MetalBuffer pred, uint32_t step_k, uint32_t step_j,
                                 uint32_t n) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"sort_bitonic_swap_idx");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)idx offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)pred offset:0 atIndex:1];
        [enc setBytes:&step_k length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&step_j length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, n);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Gather ─────────────────────────────────────────────────────────────────

int metal_gather(MetalBuffer operand, MetalBuffer indices, MetalBuffer dst,
                 MetalBuffer config, uint32_t total_elements,
                 uint32_t config_size, uint32_t elem_size) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"gather_bytes");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)operand offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst     offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)config  offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&elem_size length:sizeof(uint32_t) atIndex:5];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

static NSString* scatter_elem_suffix(int sk) {
    switch (sk) {
        case 0: return @"_f32";
        case 1: return @"_i32";
        case 2: return @"_u32";
        case 3: return @"_i64_serial";
        case 4: return @"_u64_serial";
        default: return @"";
    }
}

static size_t scatter_elem_size(int sk) {
    switch (sk) {
        case 0:
        case 1:
        case 2:
            return 4;
        case 3:
        case 4:
            return 8;
        default:
            return 0;
    }
}

static int scatter_cb_commit_wait(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    return ([cb error] != nil) ? -5 : 0;
}

static int scatter_stage_finish(id<MTLCommandBuffer> cb) {
    if (g_encode_cb != nil && cb == g_encode_cb) return 0;
    return scatter_cb_commit_wait(cb);
}

// Parallel int64/uint64 scatter: radix sort + segmented scan + parallel writes.
// is_u64: 0=i64, 1=u64. reduce_op: 0=sum, 1=max, 2=min.
static int metal_run_scatter_64_fast(
    MetalBuffer operand, MetalBuffer indices, MetalBuffer updates, MetalBuffer dst,
    MetalBuffer config, uint32_t n_updates, int is_u64, int reduce_op) {
    if (reduce_op < 0 || reduce_op > 2 || (is_u64 != 0 && is_u64 != 1))
        return -1;

    id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst;
    id<MTLBuffer> opBuf = (__bridge id<MTLBuffer>)operand;
    size_t dstLen = [dstBuf length];
    if (dstLen < 8 || dstLen % 8 != 0) return -1;
    uint32_t numel = (uint32_t)(dstLen / 8);

    uint32_t npad = 1u;
    while (npad < n_updates) npad <<= 1;
    if (npad > (1u << 24))
        return -1;
    uint32_t pad_key_base = numel + npad + 16u;

    id<MTLBuffer> k0 = [g_device newBufferWithLength:npad * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> v0 = [g_device newBufferWithLength:npad * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> k1 = [g_device newBufferWithLength:npad * sizeof(uint32_t)
                                             options:MTLResourceStorageModeShared];
    id<MTLBuffer> v1 = [g_device newBufferWithLength:npad * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> sortKey = [g_device newBufferWithLength:npad * sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];
    id<MTLBuffer> sortPerm = [g_device newBufferWithLength:npad * sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> sa = [g_device newBufferWithLength:npad * 8u options:MTLResourceStorageModeShared];
    id<MTLBuffer> sb = [g_device newBufferWithLength:npad * 8u options:MTLResourceStorageModeShared];
    if (!k0 || !v0 || !k1 || !v1 || !sortKey || !sortPerm || !sa || !sb) return -1;

    memcpy([dstBuf contents], [opBuf contents], dstLen);
    memset([k0 contents], 0, npad * sizeof(uint32_t));
    memset([v0 contents], 0, npad * 8u);

    NSString* knProj = is_u64 ? @"scatter_u64_proj_keys" : @"scatter_i64_proj_keys";
    NSString* knPad = is_u64 ? @"scatter_pad_dummy_keys_u64" : @"scatter_pad_dummy_keys_i64";
    NSString* knGather = is_u64 ? @"scatter_gather_pair_u64" : @"scatter_gather_pair_i64";
    NSString* knScan = nil;
    NSString* knApply = nil;
    if (!is_u64) {
        if (reduce_op == 0) {
            knScan = @"scatter_seg_scan_i64_add";
            knApply = @"scatter_i64_apply_sum";
        } else if (reduce_op == 1) {
            knScan = @"scatter_seg_scan_i64_max";
            knApply = @"scatter_i64_apply_max";
        } else {
            knScan = @"scatter_seg_scan_i64_min";
            knApply = @"scatter_i64_apply_min";
        }
    } else {
        if (reduce_op == 0) {
            knScan = @"scatter_seg_scan_u64_add";
            knApply = @"scatter_u64_apply_sum";
        } else if (reduce_op == 1) {
            knScan = @"scatter_seg_scan_u64_max";
            knApply = @"scatter_u64_apply_max";
        } else {
            knScan = @"scatter_seg_scan_u64_min";
            knApply = @"scatter_u64_apply_min";
        }
    }

    id<MTLComputePipelineState> pipeProj = get_pipeline(knProj);
    id<MTLComputePipelineState> pipePad = get_pipeline(knPad);
    id<MTLComputePipelineState> pipeRadixPrep = get_pipeline(@"scatter_radix_digit_sort_prepare");
    id<MTLComputePipelineState> pipeBitonic = get_pipeline(@"scatter_bitonic_step_u32");
    id<MTLComputePipelineState> pipeGather = get_pipeline(knGather);
    id<MTLComputePipelineState> pipeScan = get_pipeline(knScan);
    id<MTLComputePipelineState> pipeApply = get_pipeline(knApply);
    if (!pipeProj || !pipePad || !pipeRadixPrep || !pipeBitonic || !pipeGather || !pipeScan ||
        !pipeApply)
        return -1;

    {
        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeProj];
        [enc setBuffer:k0 offset:0 atIndex:0];
        [enc setBuffer:v0 offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)updates offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:4];
        [enc setBytes:&n_updates length:sizeof(uint32_t) atIndex:5];
        dispatch_1d(pipeProj, enc, n_updates);
        [enc endEncoding];

        enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipePad];
        [enc setBuffer:k0 offset:0 atIndex:0];
        [enc setBuffer:v0 offset:0 atIndex:1];
        [enc setBytes:&n_updates length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&npad length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&pad_key_base length:sizeof(uint32_t) atIndex:4];
        dispatch_1d(pipePad, enc, npad);
        [enc endEncoding];
        if (scatter_stage_finish(cb) != 0) return -5;
    }

    id<MTLBuffer> kIn = k0;
    id<MTLBuffer> kOut = k1;
    id<MTLBuffer> vIn = v0;
    id<MTLBuffer> vOut = v1;

    for (int pass = 0; pass < 4; pass++) {
        uint32_t shift = (uint32_t)(pass * 8);

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setComputePipelineState:pipeRadixPrep];
        [enc setBuffer:kIn offset:0 atIndex:0];
        [enc setBuffer:sortKey offset:0 atIndex:1];
        [enc setBuffer:sortPerm offset:0 atIndex:2];
        [enc setBytes:&shift length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&npad length:sizeof(uint32_t) atIndex:4];
        dispatch_1d(pipeRadixPrep, enc, npad);

        for (uint32_t k = 2u; k <= npad; k <<= 1) {
            for (uint32_t j = k >> 1; j > 0u; j >>= 1) {
                [enc setComputePipelineState:pipeBitonic];
                [enc setBuffer:sortKey offset:0 atIndex:0];
                [enc setBuffer:sortPerm offset:0 atIndex:1];
                [enc setBytes:&k length:sizeof(uint32_t) atIndex:2];
                [enc setBytes:&j length:sizeof(uint32_t) atIndex:3];
                [enc setBytes:&npad length:sizeof(uint32_t) atIndex:4];
                dispatch_1d(pipeBitonic, enc, npad);
            }
        }

        [enc setComputePipelineState:pipeGather];
        [enc setBuffer:kIn offset:0 atIndex:0];
        [enc setBuffer:vIn offset:0 atIndex:1];
        [enc setBuffer:sortPerm offset:0 atIndex:2];
        [enc setBuffer:kOut offset:0 atIndex:3];
        [enc setBuffer:vOut offset:0 atIndex:4];
        [enc setBytes:&npad length:sizeof(uint32_t) atIndex:5];
        dispatch_1d(pipeGather, enc, npad);
        [enc endEncoding];
        if (scatter_stage_finish(cb) != 0) return -5;

        id<MTLBuffer> kt = kIn;
        kIn = kOut;
        kOut = kt;
        id<MTLBuffer> vt = vIn;
        vIn = vOut;
        vOut = vt;
    }

    id<MTLBuffer> kSorted = kIn;
    id<MTLBuffer> vSorted = vIn;

    id<MTLBuffer> inBuf = vSorted;
    id<MTLBuffer> outBuf = sa;
    for (uint32_t stride = 1u; stride < npad; stride <<= 1u) {
        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeScan];
        [enc setBuffer:kSorted offset:0 atIndex:0];
        [enc setBuffer:inBuf offset:0 atIndex:1];
        [enc setBuffer:outBuf offset:0 atIndex:2];
        [enc setBytes:&stride length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&npad length:sizeof(uint32_t) atIndex:4];
        dispatch_1d(pipeScan, enc, npad);
        [enc endEncoding];
        if (scatter_stage_finish(cb) != 0) return -5;
        id<MTLBuffer> tmp = inBuf;
        inBuf = outBuf;
        outBuf = tmp;
    }
    id<MTLBuffer> vFinal = inBuf;

    {
        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeApply];
        [enc setBuffer:dstBuf offset:0 atIndex:0];
        [enc setBuffer:kSorted offset:0 atIndex:1];
        [enc setBuffer:vFinal offset:0 atIndex:2];
        [enc setBytes:&numel length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&npad length:sizeof(uint32_t) atIndex:4];
        dispatch_1d(pipeApply, enc, npad);
        [enc endEncoding];
        if (scatter_stage_finish(cb) != 0) return -5;
    }
    return 0;
}

static const uint32_t kScatter64FastMinUpdates = 2048u;

// ─── ScatterSum ─────────────────────────────────────────────────────────────

int metal_scatter_sum(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem) {
    (void)config_size;
    @autoreleasepool {
        size_t es = scatter_elem_size(scatter_elem);
        if (es == 0 || [(__bridge id<MTLBuffer>)dst length] % es != 0) return -1;

        if (scatter_elem == 3 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 0,
                                           0) == 0) {
                return 0;
            }
        }
        if (scatter_elem == 4 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 1,
                                           0) == 0) {
                return 0;
            }
        }

        NSString* kernel_name = [NSString stringWithFormat:@"scatter_sum%@",
                                 scatter_elem_suffix(scatter_elem)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLBuffer> opBuf  = (__bridge id<MTLBuffer>)operand;
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst;
        size_t opLen = [opBuf length];
        memcpy([dstBuf contents], [opBuf contents], opLen);

        uint32_t dispatch_threads = total_elements;
        if (scatter_elem == 3 || scatter_elem == 4) {
            dispatch_threads = 1;
        }

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)dst     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)updates offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)config  offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, dispatch_threads);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_scatter_max(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem) {
    (void)config_size;
    @autoreleasepool {
        size_t es = scatter_elem_size(scatter_elem);
        if (es == 0 || [(__bridge id<MTLBuffer>)dst length] % es != 0) return -1;

        if (scatter_elem == 3 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 0,
                                           1) == 0) {
                return 0;
            }
        }
        if (scatter_elem == 4 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 1,
                                           1) == 0) {
                return 0;
            }
        }

        NSString* kernel_name = [NSString stringWithFormat:@"scatter_max%@",
                                 scatter_elem_suffix(scatter_elem)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLBuffer> opBuf  = (__bridge id<MTLBuffer>)operand;
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst;
        memcpy([dstBuf contents], [opBuf contents], [opBuf length]);

        uint32_t dispatch_threads = total_elements;
        if (scatter_elem == 3 || scatter_elem == 4) {
            dispatch_threads = 1;
        }

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)dst     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)updates offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)config  offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, dispatch_threads);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_scatter_min(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem) {
    (void)config_size;
    @autoreleasepool {
        size_t es = scatter_elem_size(scatter_elem);
        if (es == 0 || [(__bridge id<MTLBuffer>)dst length] % es != 0) return -1;

        if (scatter_elem == 3 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 0,
                                           2) == 0) {
                return 0;
            }
        }
        if (scatter_elem == 4 && total_elements >= kScatter64FastMinUpdates) {
            if (metal_run_scatter_64_fast(operand, indices, updates, dst, config, total_elements, 1,
                                           2) == 0) {
                return 0;
            }
        }

        NSString* kernel_name = [NSString stringWithFormat:@"scatter_min%@",
                                 scatter_elem_suffix(scatter_elem)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLBuffer> opBuf  = (__bridge id<MTLBuffer>)operand;
        id<MTLBuffer> dstBuf = (__bridge id<MTLBuffer>)dst;
        memcpy([dstBuf contents], [opBuf contents], [opBuf length]);

        uint32_t dispatch_threads = total_elements;
        if (scatter_elem == 3 || scatter_elem == 4) {
            dispatch_threads = 1;
        }

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)dst     offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)indices offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)updates offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)config  offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, dispatch_threads);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── ConvGeneral ────────────────────────────────────────────────────────────

int metal_conv_general(MetalBuffer input, MetalBuffer kernel_buf, MetalBuffer dst,
                       MetalBuffer config, uint32_t total_elements,
                       uint32_t config_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"conv_general%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)input      offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)kernel_buf offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst        offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)config     offset:0 atIndex:3];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:4];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── ReduceWindow ───────────────────────────────────────────────────────────

int metal_reduce_window(MetalBuffer src, MetalBuffer dst,
                        MetalBuffer config, uint32_t total_elements,
                        uint32_t config_size, int dtype) {
    @autoreleasepool {
        NSString* kernel_name = [NSString stringWithFormat:@"reduce_window%@",
                                 dtype_suffix(dtype)];
        id<MTLComputePipelineState> pipeline = get_pipeline(kernel_name);
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBuffer:(__bridge id<MTLBuffer>)src    offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst    offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)config offset:0 atIndex:2];
        [enc setBytes:&total_elements length:sizeof(uint32_t) atIndex:3];

        dispatch_1d(pipeline, enc, total_elements);
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── RNG (PCG, Go-compatible) ────────────────────────────────────────────────

int metal_rng_pcg_fill(MetalBuffer state_in, MetalBuffer state_out, MetalBuffer dst,
                       uint32_t num_bytes) {
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = get_pipeline(@"rng_pcg_fill_bytes");
        if (!pipeline) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pipeline];

        [enc setBuffer:(__bridge id<MTLBuffer>)state_in  offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)state_out offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)dst       offset:0 atIndex:2];
        [enc setBytes:&num_bytes length:sizeof(uint32_t) atIndex:3];

        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── Batch norm training / gradient (GPU kernels) ───────────────────────────

static NSString* bn_kernel_suffix(int elem_dtype) {
    return (elem_dtype == 0) ? @"_f16" : @"";
}

int metal_bn_training_forward(
    MetalBuffer x, MetalBuffer scale, MetalBuffer offset,
    MetalBuffer out_norm, MetalBuffer out_mean, MetalBuffer out_var,
    MetalBnGeom geom, float epsilon, int elem_dtype) {
    @autoreleasepool {
        NSString* ks = bn_kernel_suffix(elem_dtype);
        id<MTLComputePipelineState> pipe_mean =
            get_pipeline([NSString stringWithFormat:@"bn_train_reduce_mean%@", ks]);
        id<MTLComputePipelineState> pipe_var =
            get_pipeline([NSString stringWithFormat:@"bn_train_reduce_var%@", ks]);
        id<MTLComputePipelineState> pipe_norm =
            get_pipeline([NSString stringWithFormat:@"bn_train_normalize%@", ks]);
        if (!pipe_mean || !pipe_var || !pipe_norm) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        MetalBnGeom gcopy = geom;

        [enc setComputePipelineState:pipe_mean];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_mean offset:0 atIndex:1];
        [enc setBytes:&gcopy length:sizeof(MetalBnGeom) atIndex:2];
        NSUInteger groups = geom.channels > 0 ? geom.channels : 1;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:pipe_var];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_mean offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_var offset:0 atIndex:2];
        [enc setBytes:&gcopy length:sizeof(MetalBnGeom) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:pipe_norm];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)offset offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_mean offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_var offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)out_norm offset:0 atIndex:5];
        [enc setBytes:&gcopy length:sizeof(MetalBnGeom) atIndex:6];
        [enc setBytes:&epsilon length:sizeof(float) atIndex:7];
        dispatch_1d(pipe_norm, enc, geom.numel);

        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

int metal_bn_gradient(
    MetalBuffer x, MetalBuffer scale, MetalBuffer mean, MetalBuffer var,
    MetalBuffer dy, MetalBuffer dx, MetalBuffer dgamma, MetalBuffer dbeta,
    MetalBnGeom geom, float epsilon, int elem_dtype) {
    @autoreleasepool {
        NSString* ks = bn_kernel_suffix(elem_dtype);
        id<MTLComputePipelineState> pipe_red =
            get_pipeline([NSString stringWithFormat:@"bn_grad_reduce%@", ks]);
        id<MTLComputePipelineState> pipe_dx =
            get_pipeline([NSString stringWithFormat:@"bn_grad_dx%@", ks]);
        if (!pipe_red || !pipe_dx) return -1;

        id<MTLCommandBuffer> cb = (g_encode_cb != nil) ? g_encode_cb : [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        MetalBnGeom gcopy = geom;
        NSUInteger groups = geom.channels > 0 ? geom.channels : 1;

        [enc setComputePipelineState:pipe_red];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)dy offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)dgamma offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)dbeta offset:0 atIndex:5];
        [enc setBytes:&gcopy length:sizeof(MetalBnGeom) atIndex:6];
        [enc setBytes:&epsilon length:sizeof(float) atIndex:7];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

        [enc setComputePipelineState:pipe_dx];
        [enc setBuffer:(__bridge id<MTLBuffer>)x offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)scale offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)mean offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)var offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)dy offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)dbeta offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)dgamma offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)dx offset:0 atIndex:7];
        [enc setBytes:&gcopy length:sizeof(MetalBnGeom) atIndex:8];
        [enc setBytes:&epsilon length:sizeof(float) atIndex:9];
        dispatch_1d(pipe_dx, enc, geom.numel);

        [enc endEncoding];
        if (g_encode_cb != nil && cb == g_encode_cb) return 0;
        return commit_and_wait(cb);
    }
}

// ─── AdamW step ─────────────────────────────────────────────────────────────
// Deliberate stub: optimizer updates are applied at the graph level in GoMLX.

int metal_adamw_step(MetalBuffer param, MetalBuffer grad,
                     MetalBuffer m_buf, MetalBuffer v_buf, MetalBuffer dst,
                     uint32_t num_elements, float lr, float beta1, float beta2,
                     float epsilon, float weight_decay, int step, int dtype) {
    fprintf(stderr, "gomlx-metal: metal_adamw_step is unsupported; optimizers are handled at the graph level in GoMLX\n");
    return -5;
}
