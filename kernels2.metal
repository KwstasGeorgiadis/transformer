#include <metal_stdlib>
using namespace metal;

kernel void gpu_add(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = A[id] + B[id];
}

kernel void gpu_dot(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* result_mat[[buffer(2)]],
    constant uint& M [[buffer(3)]],    // Rows of A
    constant uint& K [[buffer(4)]],    // Cols of A = Rows of B
    constant uint& N [[buffer(5)]],    // Cols of B
    uint2 gid[[thread_position_in_grid]]
)
{
    uint i = gid.y; //row
    uint j = gid.x; //column 
    
    if (i >= M || j >= N) return;
    float sum = 0.0;
    for(uint k = 0; k < K; k++){
        sum+=A[i*K+k] * B[k*N+j];
    }
    result_mat[i*N+j] = sum;
    
}

//col[0]col[1]col[2]col[3]
// a00  a01  a02  a03  row[0]
//a10  a11  a12  a13  row[1]
//a20  a21  a22  a23  row[2]
//M = 3 K = 4 (3,4)

//col[0]col[1]
//b00  b01row[0]
//b10  b11row[1]
//b20  b21row[2]
//b30  b31row[3]
//K=4 N=2 (4,2)

//if C = A times B and A is (M,K) and B is (K,N) C is (M,N)

kernel void gpu_substract(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]){

        result[id] = A[id] - B[id];
    }

kernel void gpu_relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]){
        output[id] = max(0.0f,[id];)
    }

kernel void gpu_relu_derivative(
    device const float* input [[buffer(0)]],
    device float* output[[buffer(1)]],
    uint id [[thread_position_in_grid]])
    {
        output[id] = input[id] > 0.0f ? 1.0f :0.0f;
    }

kernel void gpu_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device uint& cols [[buffer(2)]]
    uint row [[thread_position_in_grid]]){

        float max_val = input[row *cols]
        for (uint i=0, i<cols ;i++){
            float exp_val= exp(input[row * cols + i] - max_val); 
            output [row * col+i] = exp_val;
            sum += exp_val;
        }
        for (uint i = 0; i < cols; i++) {
            output[row * cols + i] /= sum;
        }

}
kernel void gpu_exp(
    device const float* input [[buffer(0)]],
    device float* output[[buffer(1)]],
    uint id[[thread_position_in_grid]])
    {
        output[id] = exp(input[id]);
    }

kernel void gpu_square(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) 
    {
        output[id] = input[id] * input[id];
    }


kernel void gpu_embedding_lookup(
    device const float* embed [[buffer(0)]],
    device const uint* indices[[buffer(1)]],
    device float* output[[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& embed_dim[[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
    {
        uint seq_idx = gid.x ;
        uint batch_idx = gid.y;

        uint pos = batch_idx * seq_len * seq_idx;
        uint token_id = indices[pos];

        uint out_base = pos * embed_dim ; 
        uint emb_base = token_id * embed_dim;

        for(uint i = 0;i < embed_dim;i++){
            output[out_base + i] = embed_table[emb_base + i];
        } 

    }



kernel void gpu_sqrt(
    device const float* input[[buffer(0)]],
    device float* output[[buffer(1)]],
    uint id [[thread_position_in_grid]])
    {
        output[id] = sqrt(input[id])
    }

kernel void gpu_square(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = input[id] * input[id];
}

kernel void gpu_scale(
    device const float* A [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    result[id] = A[id] * scale;
}

kernel void gpu_add_scalar(
    device const float* A [[buffer(0)]],
    device float* result [[buffer(1)]],
    constant float& scalar [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    result[id] = A[id] + scalar;
}

kernel void gpu_divide(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* result[[buffer(2)]]
    uint id [[thread_position_in_grid]])
    {
        result[id] = A[id] / B[id]
    }

kernel void gpu_layer_norm(
    device const float* input[[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device const float* variance [[buffer(2)]],
    device const float* gamma [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& last_dim [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]])
    {
        uint group_idx = id / last_dim;
        uint dim_idx = id % last_dim;

        float norm = (input[id] - mean[group_idx]) / sqrt(variance[group_idx] + eps);
        output[id] = gamma[dim_idx] * norm + beta[dim_idx]; 

    }
kernel void gpu_matmul_add_bias(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.y;
    uint j = gid.x;
    
    if (i >= M || j >= N) return;
    
    float sum = 0.0;
    for(uint k = 0; k < K; k++) {
        sum += input[i*K + k] * weights[k*N + j];
    }
    output[i*N + j] = sum + bias[j];
}

kernel void gpu_mean_last_dim(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& total_elements [[buffer(2)]],
    constant uint& last_dim [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    uint num_groups = total_elements / last_dim;
    if (idx >= num_groups) return;
    
    float sum = 0.0f;
    uint base = idx * last_dim;
    for (uint i = 0; i < last_dim; i++) {
        sum += input[base + i];
    }
    output[idx] = sum / float(last_dim);
}

kernel void gpu_variance_last_dim(
    device const float* input [[buffer(0)]],
    device const float* mean [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& last_dim [[buffer(4)]],
    uint idx [[thread_position_in_grid]]
) {
    uint num_groups = total_elements / last_dim;
    if (idx >= num_groups) return;
    
    float sum_sq = 0.0f;
    uint base = idx * last_dim;
    float mean_val = mean[idx];
    
    for (uint i = 0; i < last_dim; i++) {
        float diff = input[base + i] - mean_val;
        sum_sq += diff * diff;
    }
    output[idx] = sum_sq / float(last_dim);
}
