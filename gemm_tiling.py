# -*- coding: utf-8 -*-
import math
from tabulate import tabulate

# constants
GB = 1024*1024*1024
MB = 1024*1024
KB = 1024
ns = 1e-9
us = 1e-6
ms = 1e-3
# 1 TFLOPS = 1e12 FLOPS 还是 2**40 FLOPS （1.0995e12)? 这个影响还是有一些的
TFLOPS = 1e12
# TFLOPS = 2 ** 40


# hardware configuration
data_type = 2  # 2 bytes for FP16, 4 bytes for FP32
Tile_num = 4 * 4  # 4x4 tiles
Tile_SRAM = 3 * MB  # 3MB
Tile_compute = 128/16 * TFLOPS # 8 TFLOPS
DDR_BW = 100 * GB  # 100GB/s
NOC_BW = 128 * GB  # 128GB/s
NOC_latency_hop = 10 * ns  # 10ns for 1 hop
DDR_latency = 100 * ns  # 100ns

def gemm_tiling_input_stationary(B, M, K, N, tile_m, tile_n, print_details=True):
    """analytical perf of gemm tiling with input stationary
    double buffer for input, triple buffer for weight, double buffer for output 
    Args:
        B (int): batch size
        M (int): left matrix size: (M, K)
        K (int): 
        N (int): right matrix size: (K, N)
        tile_m (int): left matrix tile size: (tile_m, K)
        tile_n (int): right matrix tile size: (K, tile_n)
    """
    M = M * B
    input_size = tile_m * K * data_type
    weight_size = K * tile_n * data_type
    output_size = tile_m * tile_n * data_type
    compute_one_tile = tile_m * K * tile_n * 2   # 2 FLOPS per element

    input_load_time = (input_size * Tile_num / DDR_BW + DDR_latency)/ us    # us
    weight_load_time = (weight_size * Tile_num / DDR_BW + DDR_latency) / us  # us
    weight_noc_time = (weight_size / NOC_BW + NOC_latency_hop * 1) / us  # us
    output_save_time = (output_size * Tile_num / DDR_BW + DDR_latency) / us  # us
    compute_time_one_tile = compute_one_tile / Tile_compute / us  # us

    # input 需要从DDR load的次数
    input_load_iters = int(math.ceil(M / tile_m / Tile_num))
    # weight 需要从DDR load的次数
    weight_load_iters = int(math.ceil(N / tile_n / Tile_num))
    
    # 默认的buffer数量
    input_buffer_num = 2
    weight_buffer_num = 3
    output_buffer_num = 2

    # memory anylasis
    total_size = input_size*input_buffer_num+weight_size*weight_buffer_num+output_size*output_buffer_num
    if total_size > Tile_SRAM:
        print(f"Warning: using defautl buffer strategy, nedd total memory {total_size/MB:.6f} MB, > SRAM {Tile_SRAM/MB} MB")
        # 尝试减少input buffer的数量
        input_buffer_num = 1
        total_size = input_size*input_buffer_num + weight_size*weight_buffer_num + output_size*output_buffer_num
        if total_size < Tile_SRAM:
            # input 不 double buffer，每次重新load input，然后计算
            print(f"Warning: change input buffer strategy to input_buffer_num =  {input_buffer_num}")
        else:
            print(f"Error: total size without input buffer {total_size/MB:.6f} MB  is smaller than SRAM size {Tile_SRAM/MB} MB")
    
    data = [["input_size", f"{input_size/MB: .6f} * {input_buffer_num} = {input_size*input_buffer_num/MB: .6f}"], 
        ["weight_size", f"{weight_size/MB: .6f} * {weight_buffer_num} = {weight_size*weight_buffer_num/MB: .6f}"], 
        ["output_size", f"{output_size/MB: .6f} * {output_buffer_num} = {output_size*output_buffer_num/MB: .6f}"], 
        ["total_size", f"{total_size/MB: .6f}"],
        ["input_load_iters", f"{input_load_iters}"],
        ["weight_load_iters", f"{weight_load_iters}"]
    ]
    headers = ["var", "mem (MB)"]
    if print_details or total_size > Tile_SRAM:
        print("=========memory anylasis ===================")
        print(tabulate(data, headers=headers, tablefmt="pretty", ))
    if total_size > Tile_SRAM:
        return 0


    # unit time analysis
    data = [["input_load_time", f"{input_load_time: .6f}"], 
            ["weight_load_time", f"{weight_load_time: .6f}"], 
            ["weight_noc_time", f"{weight_noc_time: .6f}"], 
            ["compute_time_one_tile", f"{compute_time_one_tile: .6f}"],
            ["output_save_time", f"{output_save_time: .6f}"]
        ]
    headers = ["unit", "time (us)"]
    if print_details:
        print("=========unit time anylasis ===================")
        print(tabulate(data, headers=headers, tablefmt="pretty", ))

    # 一般save output的时间小于计算一次tile的时间，这里默认output save的时间能隐藏在计算tile的时间里
    assert output_save_time < compute_time_one_tile

    # 计算一次 noc片上流转的时间
    if compute_time_one_tile < weight_noc_time:

        time_one_noc_pipe_flow = (Tile_num-1) * weight_noc_time + compute_time_one_tile * 1
    else:
        time_one_noc_pipe_flow = Tile_num * compute_time_one_tile

    # 迭代load weight和noc流转，一次的时间选取两者最大值
    time_one_iter_w = max(weight_load_time, time_one_noc_pipe_flow)

    # 最外层迭代load input，当load一次新的input时
    # 如果double buffer input：一次的时间选取load input和noc流转的最大值
    # 否则： load input + noc流转
    if input_buffer_num == 2:
        time_one_iter_in = max(input_load_time, time_one_noc_pipe_flow)
    else:
        time_one_iter_in = input_load_time + weight_noc_time

    # internal time analysis
    data = [["time_one_noc_pipe_flow", f"{time_one_noc_pipe_flow: .6f}"], 
            ["time_one_iter_w", f"{time_one_iter_w: .6f}"], 
            ["time_one_iter_in", f"{time_one_iter_in: .6f}"],
            ["compute_time_in_one_iter", f"{Tile_num * compute_time_one_tile: .6f}"]
        ]
    headers = ["item", "time (us)"]
    if print_details:
        print("=========internal time anylasis ===================")
        print(tabulate(data, headers=headers, tablefmt="pretty", ))


    inital_load_time = input_load_time + weight_load_time
    iter_over_weight_time = (weight_load_iters - 1) * time_one_iter_w * input_load_iters
    iter_over_input_time = (input_load_iters - 1) * time_one_iter_in
    last_iter_time = time_one_noc_pipe_flow
    total_time = inital_load_time + iter_over_weight_time + iter_over_input_time + last_iter_time + output_save_time
    total_compute_time = M * K * N * 2 / Tile_compute / Tile_num / us
    utilization = total_compute_time / total_time * 100  # %
    # final result analysis
    data = [["inital_load_time", f"{inital_load_time: .6f}", f"{inital_load_time/total_time*100: .2f}%"], 
            ["iter_over_weight_time", f"{iter_over_weight_time: .6f}", f"{iter_over_weight_time/total_time*100: .2f}%"], 
            ["iter_over_input_time", f"{iter_over_input_time: .6f}", f"{iter_over_input_time/total_time*100: .2f}%"],
            ["last_iter_time", f"{last_iter_time: .6f}", f"{last_iter_time/total_time*100: .2f}%"],
            ["total_time", f"{total_time: .6f}", "100%"],
            ["total_compute_time", f"{total_compute_time: .6f}", f"{utilization: .2f}%"],
            ["utilization", f"{utilization: .2f}%"]
        ]
    headers = ["item", "time (us)", "percentage"]
    if print_details:
        print("=========final time anylasis ===================")
        print(tabulate(data, headers=headers, tablefmt="pretty"))
    return  utilization

    
def gemm_tiling_weight_stationary(B, M, K, N, tile_m, tile_n, print_details=True):
    """analytical perf of gemm tiling with weight stationary
    double buffer for weight , triple buffer for input, double buffer for output;
    for simple, we only need transpose the input matrix and weitght matrix, 
    use the weight matrix as input, call input stationary function
    Args:
        B (int): batch size
        M (int): left matrix size: (M, K)
        K (int): 
        N (int): right matrix size: (K, N)
        tile_m (int): left matrix tile size: (tile_m, K)
        tile_n (int): right matrix tile size: (K, tile_n)
    """
    M = M * B
    tile_m, tile_n = tile_n, tile_m
    M, K, N = N, K, M
    B = 1
    return gemm_tiling_input_stationary(B, M, K, N, tile_m, tile_n, print_details)


if __name__ == "__main__":
    # =============== QKV projection  ===================
    M, K , N = 4096, 4096, 4096
    QKV = 1   # QKV = 3, if fuse q k v into one matrix
    N = N * QKV
    B = 1

    # tile_m = 64
    # tile_n = 64

    # tile_m = 128
    # tile_n = 32

    tile_m = 256
    tile_n = 32
    utilization = gemm_tiling_input_stationary(B, M, K, N, tile_m, tile_n, print_details=True)
    print(f"QKV projection, M={M}, K={K}, N={N}, B={B}, tile_m={tile_m}, tile_n={tile_n}, stationary: input, utilization={utilization:.2f}%")