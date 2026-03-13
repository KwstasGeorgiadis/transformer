#include <metal_stdlib>
using namespace metal;

// ============================================
// Matrix Operations
// ============================================

kernel void gpu_dot(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    output[row * N + col] = sum;
}

kernel void gpu_batched_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_heads [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& N [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    uint batch_head = gid.z;
    
    if (col >= N || row >= M || batch_head >= batch_size * num_heads) return;
    
    uint batch = batch_head / num_heads;
    uint head = batch_head % num_heads;
    
    uint a_offset = batch * num_heads * M * K + head * M * K;
    uint b_offset = batch * num_heads * K * N + head * K * N;
    uint out_offset = batch * num_heads * M * N + head * M * N;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[a_offset + row * K + k] * B[b_offset + k * N + col];
    }
    output[out_offset + row * N + col] = sum;
}

// ============================================
// Transpose Operations
// ============================================

kernel void gpu_transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& M [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    output[col * M + row] = input[row * N + col];
}

kernel void gpu_transpose_4d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& d0 [[buffer(2)]],
    constant uint& d1 [[buffer(3)]],
    constant uint& d2 [[buffer(4)]],
    constant uint& d3 [[buffer(5)]],
    constant uint& mode [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    uint total = d0 * d1 * d2 * d3;
    if (id >= total) return;
    
    uint i0 = id / (d1 * d2 * d3);
    uint rem = id % (d1 * d2 * d3);
    uint i1 = rem / (d2 * d3);
    rem = rem % (d2 * d3);
    uint i2 = rem / d3;
    uint i3 = rem % d3;
    
    uint out_idx;
    if (mode == 0) {
        out_idx = i0 * (d2 * d1 * d3) + i2 * (d1 * d3) + i1 * d3 + i3;
    } else {
        out_idx = i0 * (d1 * d3 * d2) + i1 * (d3 * d2) + i3 * d2 + i2;
    }
    
    output[out_idx] = input[id];
}

// ============================================
// Embedding
// ============================================

kernel void gpu_embedding_lookup(
    device const float* embed [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& num_positions [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= num_positions || col >= embed_dim) return;
    
    uint token = indices[row];
    output[row * embed_dim + col] = embed[token * embed_dim + col];
}

// ============================================
// Activations
// ============================================

kernel void gpu_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = max(0.0f, input[id]);
}

kernel void gpu_relu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    grad_in[id] = input[id] > 0.0f ? grad_out[id] : 0.0f;
}

kernel void gpu_gelu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float cdf = 0.5f * (1.0f + tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[id] = x * cdf;
}

kernel void gpu_gelu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    float x = input[id];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float tanh_inner = tanh(inner);
    float cdf = 0.5f * (1.0f + tanh_inner);
    float pdf = 0.5f * 0.7978845608f * (1.0f + 0.134145f * x * x) * (1.0f - tanh_inner * tanh_inner);
    grad_in[id] = grad_out[id] * (cdf + x * pdf);
}

// ============================================
// Softmax
// ============================================

kernel void gpu_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= rows) return;
    
    uint offset = id * cols;
    
    float max_val = input[offset];
    for (uint i = 1; i < cols; i++) {
        max_val = max(max_val, input[offset + i]);
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < cols; i++) {
        sum += exp(input[offset + i] - max_val);
    }
    
    for (uint i = 0; i < cols; i++) {
        output[offset + i] = exp(input[offset + i] - max_val) / sum;
    }
}

kernel void gpu_softmax_4d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& num_heads [[buffer(3)]],
    constant uint& seq1 [[buffer(4)]],
    constant uint& seq2 [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint batch = gid.x;
    uint head = gid.y;
    uint row = gid.z;
    
    if (batch >= batch_size || head >= num_heads || row >= seq1) return;
    
    uint base = batch * num_heads * seq1 * seq2 + head * seq1 * seq2 + row * seq2;
    
    float max_val = input[base];
    for (uint i = 1; i < seq2; i++) {
        max_val = max(max_val, input[base + i]);
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < seq2; i++) {
        sum += exp(input[base + i] - max_val);
    }
    
    for (uint i = 0; i < seq2; i++) {
        output[base + i] = exp(input[base + i] - max_val) / sum;
    }
}

kernel void gpu_softmax_backward_4d(
    device const float* softmax_out [[buffer(0)]],
    device const float* grad_out [[buffer(1)]],
    device float* grad_in [[buffer(2)]],
    constant uint& d0 [[buffer(3)]],
    constant uint& d1 [[buffer(4)]],
    constant uint& d2 [[buffer(5)]],
    constant uint& d3 [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    uint batch_head = gid.z;
    
    if (col >= d3 || row >= d2 || batch_head >= d0 * d1) return;
    
    uint base = batch_head * d2 * d3 + row * d3;
    
    float dot_sum = 0.0f;
    for (uint i = 0; i < d3; i++) {
        dot_sum += grad_out[base + i] * softmax_out[base + i];
    }
    
    uint idx = base + col;
    grad_in[idx] = softmax_out[idx] * (grad_out[idx] - dot_sum);
}

// ============================================
// Layer Normalization
// ============================================

kernel void gpu_layer_norm_forward(
    device const float* input [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* x_norm [[buffer(4)]],
    device float* variance [[buffer(5)]],
    constant uint& rows [[buffer(6)]],
    constant uint& cols [[buffer(7)]],
    constant float& eps [[buffer(8)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= rows) return;
    
    uint offset = id * cols;
    
    float mean = 0.0f;
    for (uint i = 0; i < cols; i++) {
        mean += input[offset + i];
    }
    mean /= float(cols);
    
    float var = 0.0f;
    for (uint i = 0; i < cols; i++) {
        float diff = input[offset + i] - mean;
        var += diff * diff;
    }
    var /= float(cols);
    variance[id] = var;
    
    float inv_std = 1.0f / sqrt(var + eps);
    for (uint i = 0; i < cols; i++) {
        float norm = (input[offset + i] - mean) * inv_std;
        x_norm[offset + i] = norm;
        output[offset + i] = gamma[i] * norm + beta[i];
    }
}

kernel void gpu_layer_norm_backward_dx(
    device const float* grad_out [[buffer(0)]],
    device const float* x_norm [[buffer(1)]],
    device const float* variance [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device float* grad_in [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& cols [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= rows) return;
    
    uint offset = id * cols;
    float inv_std = 1.0f / sqrt(variance[id] + eps);
    float N_f = float(cols);
    
    // Full layer norm backward:
    // dx_i = (1/std) * (1/N) * (N * dy_hat_i - sum(dy_hat) - x_norm_i * sum(dy_hat * x_norm))
    // where dy_hat_i = grad_out_i * gamma_i
    
    float sum_dy_hat = 0.0f;
    float sum_dy_hat_xn = 0.0f;
    for (uint i = 0; i < cols; i++) {
        float dy_hat = grad_out[offset + i] * gamma[i];
        sum_dy_hat += dy_hat;
        sum_dy_hat_xn += dy_hat * x_norm[offset + i];
    }
    
    for (uint i = 0; i < cols; i++) {
        float dy_hat = grad_out[offset + i] * gamma[i];
        grad_in[offset + i] = inv_std * (dy_hat - sum_dy_hat / N_f - x_norm[offset + i] * sum_dy_hat_xn / N_f);
    }
}

kernel void gpu_layer_norm_backward_gamma_beta(
    device const float* grad_out [[buffer(0)]],
    device const float* x_norm [[buffer(1)]],
    device float* dgamma [[buffer(2)]],
    device float* dbeta [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& cols [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= cols) return;
    
    float sum_dgamma = 0.0f;
    float sum_dbeta = 0.0f;
    
    for (uint r = 0; r < rows; r++) {
        uint idx = r * cols + id;
        sum_dgamma += grad_out[idx] * x_norm[idx];
        sum_dbeta += grad_out[idx];
    }
    
    dgamma[id] = sum_dgamma;
    dbeta[id] = sum_dbeta;
}

// ============================================
// Element-wise Operations
// ============================================

kernel void gpu_element_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] + b[id];
}

kernel void gpu_element_sub(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] - b[id];
}

kernel void gpu_element_mult(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = a[id] * b[id];
}

kernel void gpu_scalar_mult(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] * scalar;
}

kernel void gpu_scalar_div(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    output[id] = input[id] / scalar;
}

kernel void gpu_scale(
    device float* data [[buffer(0)]],
    constant float& scale [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    data[id] *= scale;
}

// ============================================
// Bias Operations
// ============================================

kernel void gpu_bias_add_3d(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& seq [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    uint b = gid.z;
    
    if (col >= dim || row >= seq || b >= batch) return;
    
    uint idx = b * seq * dim + row * dim + col;
    output[idx] = input[idx] + bias[col];
}

kernel void gpu_bias_add_2d(
    device const float* input [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    
    if (col >= cols || row >= rows) return;
    
    uint idx = row * cols + col;
    output[idx] = input[idx] + bias[col];
}

// ============================================
// Reduction Operations
// ============================================

kernel void gpu_sum_3d_to_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= dim) return;
    
    float sum = 0.0f;
    for (uint b = 0; b < batch; b++) {
        for (uint s = 0; s < seq; s++) {
            sum += input[b * seq * dim + s * dim + id];
        }
    }
    output[id] = sum;
}

kernel void gpu_sum_2d_to_1d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= cols) return;
    
    float sum = 0.0f;
    for (uint r = 0; r < rows; r++) {
        sum += input[r * cols + id];
    }
    output[id] = sum;
}

// ============================================
// Attention Mask
// ============================================

kernel void gpu_add_mask(
    device float* scores [[buffer(0)]],
    device const float* mask [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& seq1 [[buffer(4)]],
    constant uint& seq2 [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    uint batch_head = gid.z;
    
    if (col >= seq2 || row >= seq1 || batch_head >= batch * heads) return;
    
    uint score_idx = batch_head * seq1 * seq2 + row * seq2 + col;
    uint mask_idx = row * seq2 + col;
    
    scores[score_idx] += mask[mask_idx];
}

// ============================================
// Loss Functions
// ============================================

kernel void gpu_cross_entropy_loss(
    device const float* probs [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device float* losses [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= batch_size) return;
    
    uint label = labels[id];
    float prob = probs[id * num_classes + label];
    losses[id] = -log(prob + 1e-10f);
}

kernel void gpu_softmax_ce_backward(
    device const float* probs [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device float* grad [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    constant float& scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    
    if (col >= num_classes || row >= batch_size) return;
    
    uint idx = row * num_classes + col;
    float g = probs[idx];
    if (col == labels[row]) {
        g -= 1.0f;
    }
    grad[idx] = g * scale;
}

// ============================================
// Gradient Operations
// ============================================

kernel void gpu_scatter_add(
    device float* output [[buffer(0)]],
    device const uint* indices [[buffer(1)]],
    device const float* updates [[buffer(2)]],
    constant uint& num_indices [[buffer(3)]],
    constant uint& embed_dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint row = gid.y;
    
    if (col >= embed_dim || row >= num_indices) return;
    
    uint token_idx = indices[row];
    uint out_idx = token_idx * embed_dim + col;
    uint update_idx = row * embed_dim + col;
    
    // CAS loop for float atomic add (Metal doesn't have atomic_float)
    device atomic_uint* addr = (device atomic_uint*)&output[out_idx];
    float val = updates[update_idx];
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    uint desired;
    do {
        float current = as_type<float>(expected);
        desired = as_type<uint>(current + val);
    } while (!atomic_compare_exchange_weak_explicit(addr, &expected, desired,
              memory_order_relaxed, memory_order_relaxed));
}

// Zero a buffer
kernel void gpu_zero_buffer(
    device float* buf [[buffer(0)]],
    uint id [[thread_position_in_grid]])
{
    buf[id] = 0.0f;
}

// ============================================
// Optimizer
// ============================================

kernel void gpu_adam_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& beta1_t [[buffer(8)]],
    constant float& beta2_t [[buffer(9)]],
    uint id [[thread_position_in_grid]])
{
    float g = grad[id];
    
    float m_new = beta1 * m[id] + (1.0f - beta1) * g;
    m[id] = m_new;
    
    float v_new = beta2 * v[id] + (1.0f - beta2) * g * g;
    v[id] = v_new;
    
    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);
    
    param[id] -= lr * m_hat / (sqrt(v_hat) + epsilon);
}

// Extract last token from 3D: (batch, seq, dim) -> (batch, dim)
kernel void gpu_extract_last_token(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint b = gid.y;
    
    if (col >= dim || b >= batch) return;
    
    output[b * dim + col] = input[b * seq * dim + (seq - 1) * dim + col];
}

// Expand last token grad back to 3D (zeros elsewhere)
kernel void gpu_expand_last_token(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& seq [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint col = gid.x;
    uint s = gid.y;
    uint b = gid.z;
    
    if (col >= dim || s >= seq || b >= batch) return;
    
    uint idx = b * seq * dim + s * dim + col;
    if (s == seq - 1) {
        output[idx] = input[b * dim + col];
    } else {
        output[idx] = 0.0f;
    }
}

// Compute mean of losses buffer -> single float
kernel void gpu_mean_reduce(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (id != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < count; i++) {
        sum += input[i];
    }
    output[0] = sum / float(count);
}

// Argmax per row for accuracy
kernel void gpu_argmax_accuracy(
    device const float* probs [[buffer(0)]],
    device const uint* labels [[buffer(1)]],
    device float* correct [[buffer(2)]],
    constant uint& batch_size [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= batch_size) return;
    
    uint offset = id * num_classes;
    float max_val = probs[offset];
    uint max_idx = 0;
    for (uint i = 1; i < num_classes; i++) {
        if (probs[offset + i] > max_val) {
            max_val = probs[offset + i];
            max_idx = i;
        }
    }
    correct[id] = (max_idx == labels[id]) ? 1.0f : 0.0f;
}
