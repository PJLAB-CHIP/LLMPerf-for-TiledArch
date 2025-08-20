import json
import math
def load_config(input_path):
    # 读取json配置文件
    with open(input_path, 'r') as file:
        # 从文件中加载 JSON 数据
        config = json.load(file)
    return config

# # 读取json文件
# with open('Hardware_parameter.json', 'r') as f:
#     parameters = json.load(f)

# # 获取参数
# sram = parameters['sram_size']#一个硬件tile中SRAM的大小为3MB
# noc = parameters['noc_bandwidth']#noc带宽为128GB/s
# dram = parameters['Dram_bandwidth']#dram带宽为100GB/s
# gemm = parameters['Gemm']#算力为128/16TOPS，是每个Tile的算力
# vector = parameters['Vector']#算力为1/16TFLOPS，是每个Tile的算力
# Hops = parameters['One_Hops']#一个Hop的overlap为常数0.005us
# "DRAM_LATENCY(us)": 0.1 两次读取需要的恢复间隔

class Tx8:
    '''
    input: config (TILE_NUM, sram_size, noc_bandwidth, DRAM_BW(GB/s), Gemm, VECTOR(TOPS), One_Hops) == (tiles, sram, noc, dram, gemm, vector, Hops)
    '''
    
    def __init__(self, config) -> None:
        self.config = config
    def execute(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, details):
        # i_params = [i_size, i_flag, split_k] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
        # o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等; 对于GEMM，o_flag = i_flag * w_flag; 对于FLashAttention，确保i_flag = o_flag必须要满足这两种情况
        # w_params = [w_size, w_flag, split_k] w_size为一份输出的大小，单位为MB；w_flag为输出的总份数，例如16/32/64等
        # cp = [cp0, cp1]，其中cp0 = [cp_size0, cp_type0]; cp_size为计算量，单位为GFLOPs; cp_type为计算类型, 这里认为0为Vector，1为Gemm
        # cm_size为通信量大小，单位MB
        # cm_type
        # cm_hops为通信的最大跳数

        
        # 已删掉的部分
        # dram_size为DRAM读取数据大小，单位MB    dram_size = i_size + w_size = i_params[0] + w_params[0]
        # dram_type
        # n为流水的总次数，n与输入和输出的切分次数相关 n = (i_params[1] / self.config["4X4 tile"]) * (w_params[1] / self.config["4X4 tile"])
        # w_params中的w_cm_flag，w_params[2] = w_cm_flag，这里需要条件判断，如果w_params[1] >= self.config["4X4 tile"]，则w_cm_flag = self.config["4X4 tile"] - 1
        
        # 更改的部分
        # cp = [cp0, cp1]，其中cp0 = [cp_size0, cp_type0]
        #print(cp)
        
        Mode = self.Mode(i_params, o_params, w_params, cp)#判断是哪种模式；10代表一个Vector算子，11代表一个GEMM算子，21代表先Vector后GEMM，22代表先GEMM后Vector，31代表flash-attention
        
        verification_result, verification_flag = self.verification(i_params, o_params, w_params, cp, Mode, details)#要求(1)是否可以计算;True or False
        

        
        if Mode == 10:#一个Vector算子
            total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = self.Vector(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details)
        elif Mode == 11:#一个GEMM算子
            total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = self.GEMM(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details)
        elif Mode == 21:#先Vector，后GEMM融合
            total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = self.Vector_GEMM(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details)
        elif Mode == 22:#先GEMM，后Vector融合
            total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = self.GEMM_Vector(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details)
        else:#Mode == 31,flash-attention
            total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = self.FlashAttention(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details)
        
        
        return verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
    
    def verification(self, i_params, o_params, w_params, cp, details):
        """
        Verifies if sufficient SRAM is available for a given operation.

        This function checks if the available SRAM is sufficient to perform a computation, considering different operation types (vector or matrix) and tiling strategies.  It prioritizes maintaining multiple copies of weights in SRAM to reduce memory access overhead.

        Args:
            self: The instance of the class containing this function.  Accesses `self.config["TILE_NUM"]` and `self.config["SRAM(MB)"]` for configuration parameters.
            i_params: A list [i_size, i_flag, split_k] representing input tensor parameters.  i_size (MB): size of one input copy; i_flag: total number of input copies; split_k: indicates if the 'K' dimension is split (relevant for matrix operations).
            o_params: A list [o_size, o_flag] representing output tensor parameters. o_size (MB): size of one output copy; o_flag: total number of output copies.
            w_params: A list [w_size, w_flag, split_k] representing weight tensor parameters. w_size (MB): size of one weight copy; w_flag: total number of weight copies; split_k: same as in i_params.
            cp: A list [[cp_size0, cp_type0]] representing computation parameters. cp_size0 (GFLOPs): computational cost; cp_type0: computation type (0: vector, 1: GEMM).  The second element of cp is currently unused.
            details: A boolean flag (default False) enabling detailed print statements (currently inactive).

        Returns:
            A tuple (verification_result, verification_flag):
                - verification_result: A boolean indicating whether sufficient SRAM is available (True) or not (False).
                - verification_flag: An integer flag. 0 indicates the standard SRAM allocation strategy was used; 1 indicates a relaxed strategy was used (e.g., using fewer input copies).
        """
    # 该函数用于验证SRAM是否够用。如果切分数量小于等于16，则不需要遵循SRAM的2-3-2分配策略（2输入，3权重，2输出） 
        o_flag = o_params[1]
        cp0 = cp[0]
        
        if len(cp) == 1 and cp0[1] == 1:
            assert i_params[2] == w_params[2]
            split_k = i_params[2]
        else:
            split_k = 1
        
        if i_params[1] == o_flag == w_params[1] and split_k == 1 and Mode == 10:#Vector算子情况，如RMSNorm
            if i_params[1] <= self.config["TILE_NUM"]:
                if i_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False
                verification_flag = 0
            else:
                if 2 * i_params[0] + 2 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                    verification_flag = 0
                else:
                    if i_params[0] + 2 * w_params[0] <= self.config["SRAM(MB)"]:
                        # if details:
                        #     print("SRAM上同时仅存在1份输入(满足条件: 1份输入 + 2份权重 < SRAM大小)")
                        verification_flag = 1
                        verification_result = True
                    else:
                        verification_result = False
                        verification_flag = 0

        else:
            if split_k == 1:
                if 2 * i_params[0] + 3 * w_params[0] + 2 * o_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                    verification_flag = 0
                else:
                    if i_params[1] == self.config["TILE_NUM"]:
                        if i_params[0] + 3 * w_params[0] + 2 * o_params[0] <= self.config["SRAM(MB)"]:
                            verification_flag = 0
                            verification_result = True
                        else:
                            verification_result = False
                            verification_flag = 0
                    else:
                        if i_params[0] + 3 * w_params[0] + 2 * o_params[0] <= self.config["SRAM(MB)"]:
                            # if details:
                            #     print("SRAM上同时仅存在1份输入(满足条件: 1份输入 + 3份权重 + 2份输出 < SRAM大小)")
                            verification_flag = 1
                            verification_result = True
                        else:
                            verification_result = False
                            verification_flag = 0
            else:
                if 2 * i_params[0] + 3 * w_params[0] + 3 * o_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                    verification_flag = 0
                else:
                    if i_params[0] + 3 * w_params[0] + 3 * o_params[0] <= self.config["SRAM(MB)"]:
                        # if details:
                        #     print("SRAM上同时仅存在1份输入(满足条件: 1份输入 + 3份权重 + 2份输出 < SRAM大小)")
                        verification_flag = 1
                        verification_result = True
                    else:
                        verification_result = False
                        verification_flag = 0

        return verification_result, verification_flag
    
    
    def Vector(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details):
        local_last = 0
        local_next = 0
        len_cp = len(cp)
        

        split_k = 1  # vector 操作不会切分 k 维度
        
        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        cp_latency_per = self.Computation_latency(cp[0])  # compute_size / TOPS
        # print(cp_latency_per)
            
        cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)  # 传输时间，每一跳的时间
        cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap  # 每一次的通信时间，单位为ms
            
        dram_store_per = 0.001 * self.config["DRAM_LATENCY(us)"] + self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        #一轮的总用时latency_cycle
        w_cm_flag = 0
        
        cp_latency_cycle = (w_cm_flag + 1) * cp_latency_per  # 每一轮的计算时间
        cm_latency_cycle = w_cm_flag * cm_latency_per  # 每一轮的通信时间，vector 操作不会通信 
        dram_read_cycle = 0.001 * self.config["DRAM_LATENCY(us)"] + \
                            self.DRAM_read(i_params, w_params, local_last, cp, len_cp)  #每一轮DRAM读取数据的时间,这部分只含权重
        dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
            
        # 计算一次 noc片上流转的时间
        if cp_latency_cycle < dram_read_cycle:  # 计算时间 < 从DRAM读取的时间 vector操作不会读weight，这种情况不会发生
            time_one_noc_pipe_flow = cm_latency_cycle + cp_latency_per * 1
        else:
            time_one_noc_pipe_flow = cp_latency_cycle
        time_one_iter_w = max(time_one_noc_pipe_flow, dram_read_cycle)
        
        input_load_time = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]
        
        if verification_flag == 0:
            time_one_iter_in = max(input_load_time, time_one_noc_pipe_flow)
        else:
            time_one_iter_in = input_load_time + cm_latency_per
        
        partial_sum_load_time = dram_store_cycle
        time_one_iter_k = max(input_load_time + partial_sum_load_time, time_one_noc_pipe_flow)        
        
        #内层循环n2
        n2 = 1
        
        cp_latency_cycle_in = n2 * cp_latency_cycle#每一个完整内层循环的计算时间
        cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            

        dram_read_cycle_in = n2 * dram_read_cycle + i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]#每一个完整内层循环的DRAM读取数据时间

        dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
        
        
        #外层循环n1
        n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))  # 切分的分数在 tile 上循环

        
        total_cp_latency = n1 * cp_latency_cycle_in * split_k
        # print(total_cp_latency)
        total_cm_latency = n1 * cm_latency_cycle_in * split_k
        total_dram_read = n1 * dram_read_cycle_in * split_k
        total_dram_store = n1 * dram_store_cycle_in * split_k
        
        total_DRAM = total_dram_read + total_dram_store + n1 * input_load_time * split_k
        
        initial_load_time = input_load_time + dram_read_cycle
        iter_over_weight_time = (n2 - 1) * time_one_iter_w * n1 * split_k
        iter_over_input_time = (n1 - 1) * time_one_iter_in
        iter_over_splitk_time = n1 * (split_k - 1) * time_one_iter_k
        last_iter_time = time_one_noc_pipe_flow
        if iter_over_weight_time + iter_over_input_time + last_iter_time + iter_over_splitk_time >= total_cp_latency:
            latency = initial_load_time + iter_over_weight_time + iter_over_input_time + iter_over_splitk_time + last_iter_time + dram_store_per
        else:
            latency = initial_load_time + total_cp_latency + dram_store_per
        # latency = initial_load_time + iter_over_weight_time + iter_over_input_time + last_iter_time + dram_store_per
        
        Utilization = total_cp_latency / latency
        
        return total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
    
    def GEMM(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details):
        local_last = 0
        local_next = 0
        len_cp = len(cp)
        
        assert i_params[2] == w_params[2]
        split_k = i_params[2]
        
        # if i_params[1] > w_params[1]:#权重驻留的情况
        #     self.swap_values(i_params, w_params)
        
        #print(i_params, w_params)
        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        cp_latency_per = self.Computation_latency(cp[0])#每一次的计算时间，单位ms
            
        cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
        cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms
            
        dram_store_per = 0.001 * self.config["DRAM_LATENCY(us)"] + self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        #一轮的总用时latency_cycle
        w_cm_flag = self.config["TILE_NUM"] - 1
        
        cp_latency_cycle = (w_cm_flag + 1) * cp_latency_per#每一轮的计算时间
        cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间
        dram_read_cycle = 0.001 * self.config["DRAM_LATENCY(us)"] + self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间,这部分只含权重
        dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
        
        # print(dram_read_cycle,dram_store_cycle)
            
        # 计算一次 noc片上流转的时间
        if cp_latency_cycle < cm_latency_cycle:
            time_one_noc_pipe_flow = cm_latency_cycle + cp_latency_per * 1
        else:
            time_one_noc_pipe_flow = cp_latency_cycle
        time_one_iter_w = max(time_one_noc_pipe_flow, dram_read_cycle)
        # print(time_one_noc_pipe_flow,dram_read_cycle,cp_latency_cycle)
        

        input_load_time = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]

        
        if verification_flag == 0:
            time_one_iter_in = max(input_load_time, time_one_noc_pipe_flow)
        else:
            time_one_iter_in = input_load_time + cm_latency_per
        
        # print(time_one_iter_in,input_load_time,time_one_noc_pipe_flow,cm_latency_per)
        
        partial_sum_load_time = dram_store_cycle
        time_one_iter_in_psum = max(input_load_time + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_w_psum = max(dram_read_cycle + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_in_k = max(dram_read_cycle + input_load_time, time_one_noc_pipe_flow)

        # print(time_one_iter_k, input_load_time,partial_sum_load_time,time_one_noc_pipe_flow)
        
        
        # #内层循环n2
        # if i_params[1] <= w_params[1]:#输入驻留
        #     n2 = w_params[1]/self.config["TILE_NUM"]
        # else:#权重驻留
        #     n2 = i_params[1]/self.config["TILE_NUM"]
        n2 = int(math.ceil(w_params[1]/self.config["TILE_NUM"]))
        
        cp_latency_cycle_in = n2 * cp_latency_cycle#每一个完整内层循环的计算时间
        #print(cp_latency_cycle_in, n2)
        
        cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            

        dram_read_cycle_in = n2 * dram_read_cycle + i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]#每一个完整内层循环的DRAM读取数据时间

            
        dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
        
        
        #外层循环n1
        # if i_params[1] <= w_params[1]:
        #     n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))
        # else:
        #     n1 = int(math.ceil(w_params[1]/self.config["TILE_NUM"]))
        n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))
        
        total_cp_latency = n1 * cp_latency_cycle_in * split_k
        total_cm_latency = n1 * cm_latency_cycle_in * split_k
        total_dram_read = n1 * dram_read_cycle_in * split_k
        total_dram_store = n1 * dram_store_cycle_in
        # print(n1,n2,cp_latency_cycle_in,cp_latency_cycle,cp_latency_per)
        # print(i_params[1])
        # print(n2,n1)
        # print(total_cp_latency,total_cm_latency,total_dram_read,total_dram_store)
        
        total_DRAM = total_dram_read + total_dram_store + n1 * input_load_time * split_k
        
        if split_k == 1:
            initial_load_time = input_load_time + dram_read_cycle
            iter_over_weight_time = (n2 - 1) * time_one_iter_w * n1
            # print(time_one_iter_w,iter_over_weight_time)
            iter_over_input_time = (n1 - 1) * time_one_iter_in
            last_iter_time = time_one_noc_pipe_flow
            if iter_over_weight_time + iter_over_input_time + last_iter_time >= total_cp_latency:
                latency = initial_load_time + iter_over_weight_time + iter_over_input_time + last_iter_time + dram_store_per
            else:
                latency = initial_load_time + total_cp_latency + dram_store_per
            # print(initial_load_time,iter_over_weight_time,iter_over_input_time,last_iter_time,dram_store_per)
        # print(iter_over_weight_time,iter_over_splitk_time,iter_over_input_time)
        # print(n1,n2)
        else:
            initial_load_time = input_load_time + dram_read_cycle
            iter_over_weight_time = (n2 - 1) * time_one_iter_w * n1
            # print(time_one_iter_w,iter_over_weight_time)
            iter_over_input_time = (n1 - 1) * time_one_iter_in
            iter_over_in_psum = (n1 - 1) * (split_k - 1) * time_one_iter_in_psum
            iter_over_w_psum = (n2 - 1) * n1 * (split_k - 1) * time_one_iter_w_psum
            iter_over_in_k = (split_k - 1) * time_one_iter_in_k
            last_iter_time = time_one_noc_pipe_flow
            if iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time >= total_cp_latency:
                latency = initial_load_time + iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time + dram_store_per
            else:
                latency = initial_load_time + total_cp_latency + dram_store_per
        
        Utilization = total_cp_latency / latency
        # print(Utilization,total_cp_latency,latency)
        
        return total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
        

    def Vector_GEMM(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details):
        local_last = 1
        local_next = 0
        len_cp = len(cp)
        
        assert i_params[2] == w_params[2]
        split_k = i_params[2]      
        # split_k = 1

        DDR_time = 0.001 * self.config["DRAM_LATENCY(us)"]
        
        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        cp_latency_per = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            cp_latency_per[i] = self.Computation_latency(cp[i])#每一个算子每一次的计算时间,由数组表示

        cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
        cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_store_per = DDR_time + self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        input_load_time = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]

    
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        #一轮的总用时latency_cycle
        cp_latency_cycle = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            cpi = cp[i]
            if cpi[1] == 0:
                cp_latency_cycle[i] = cp_latency_per[i]
            else:
                cp_latency_cycle[i] = self.config["TILE_NUM"] * cp_latency_per[i]#每一轮的计算时间
            #print(cp_latency_cycle[i])
            
        w_cm_flag = self.config["TILE_NUM"] - 1
        cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_read_cycle = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间；本身已经是一个数组了
        for i in range(len_cp):
            dram_read_cycle[i] = dram_read_cycle[i]
        dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
        
        # 计算一次 noc片上流转的时间
        if cp_latency_cycle[1] < dram_read_cycle[1]:
            time_one_noc_pipe_flow = cm_latency_cycle + cp_latency_per[1] * 1
        else:
            time_one_noc_pipe_flow = cp_latency_cycle[1]
        time_one_iter_w = max(time_one_noc_pipe_flow, dram_read_cycle[1])

        if verification_flag == 0:
            time_one_iter_in = max(input_load_time, time_one_noc_pipe_flow)
        else:
            time_one_iter_in = input_load_time + cm_latency_per
        
        partial_sum_load_time = dram_store_cycle
        time_one_iter_in_psum = max(input_load_time + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_w_psum = max(dram_read_cycle[1] + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_in_k = max(dram_read_cycle[1] + input_load_time, time_one_noc_pipe_flow)
        
        ##内层循环n2
        n2 = int(math.ceil(w_params[1]/self.config["TILE_NUM"]))

        
        cp_latency_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            if i == 1:
                cp_latency_cycle_in[i] = n2 * cp_latency_cycle[i]#每个算子每一个完整内层循环的计算时间
            else:
                cp_latency_cycle_in[i] = cp_latency_cycle[i]
            #print(cp_latency_cycle_in[i])
            
        cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            
        dram_read_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            if i == 1:
                dram_read_cycle_in[i] = n2 * dram_read_cycle[i]#每一个完整内层循环的DRAM读取数据时间;本身已经是一个数组了
            else:
                dram_read_cycle_in[i] = dram_read_cycle[i]
            
        dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
    
    
        #外层循环n1
        n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))

        
        Each_cp_latency = [0 for _ in range(len_cp)]#每一个算子的总计算时间
        total_cp_latency = 0#总计算时间
        for i in range(len_cp):
            if i == 1:
                Each_cp_latency[i] = n1 * cp_latency_cycle_in[i] * split_k
            else:
                Each_cp_latency[i] = n1 * cp_latency_cycle_in[i]
            total_cp_latency = total_cp_latency + Each_cp_latency[i]
            
        total_cm_latency = n1 * cm_latency_cycle_in * split_k#总通信时间
            
        Each_dram_read = [0 for _ in range(len_cp)]#每个算子的DRAM读取时间
        total_dram_read = 0#总DRAM读取时间
        for i in range(len_cp):
            Each_dram_read[i] = n1 * dram_read_cycle_in[i] * split_k
            total_dram_read = total_dram_read + Each_dram_read[i]
            
        total_dram_store = n1 * dram_store_cycle_in#总DRAM存储时间
        
        total_DRAM = total_dram_read + total_dram_store + n1 * input_load_time * split_k
        
        initial_load_time = input_load_time + dram_read_cycle[1]
        iter_over_weight_time = (n2 - 1) * time_one_iter_w * n1
        iter_over_input_time = (n1 - 1) * time_one_iter_in

        iter_over_in_psum = (n1 - 1) * (split_k - 1) * time_one_iter_in_psum
        iter_over_w_psum = (n2 - 1) * n1 * (split_k - 1) * time_one_iter_w_psum
        iter_over_in_k = (split_k - 1) * time_one_iter_in_k
        last_iter_time = time_one_noc_pipe_flow
        if iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time >= total_cp_latency:
            latency = initial_load_time + iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time + dram_store_per
        else:
            latency = initial_load_time + total_cp_latency + dram_store_per
        # latency = initial_load_time + iter_over_weight_time + iter_over_input_time + last_iter_time + dram_store_per
        
        Utilization = total_cp_latency / latency
        # print(Utilization,total_cp_latency,latency)
        
        return total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
        
        
    def GEMM_Vector(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details):
        local_last = 1
        local_next = 0
        len_cp = len(cp)
        
        if i_params[1] > w_params[1]:#权重驻留的情况
            self.swap_values(i_params, w_params)

        assert i_params[2] == w_params[2]
        split_k = i_params[2] 
        # split_k = 1
        
        DDR_time = 0.001 * self.config["DRAM_LATENCY(us)"]
        
        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        cp_latency_per = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            cp_latency_per[i] = self.Computation_latency(cp[i])#每一个算子每一次的计算时间,由数组表示

        cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
        cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_store_per = DDR_time + self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        input_load_time = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]

    
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        #一轮的总用时latency_cycle
        cp_latency_cycle = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            w_cm_flag = self.config["TILE_NUM"] - 1
            cp_latency_cycle[i] = (w_cm_flag + 1) * cp_latency_per[i]#每一轮的计算时间
            # print(cp_latency_cycle[i],cp_latency_per)
            
        w_cm_flag = self.config["TILE_NUM"] - 1
        cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_read_cycle = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间；本身已经是一个数组了
        for i in range(len_cp):
            dram_read_cycle[i] = dram_read_cycle[i]
        # print(dram_read_cycle)
        dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
        
        # 计算一次 noc片上流转的时间
        if cp_latency_cycle[0] < cm_latency_cycle:
            time_one_noc_pipe_flow = cm_latency_cycle + cp_latency_per[0] * 1
        else:
            time_one_noc_pipe_flow = cp_latency_cycle[0]
        time_one_iter_w = max(time_one_noc_pipe_flow, dram_read_cycle[0])
        # print(time_one_noc_pipe_flow, cm_latency_cycle,cp_latency_cycle[0])

        if verification_flag == 0:
            time_one_iter_in = max(input_load_time, time_one_noc_pipe_flow)
        else:
            time_one_iter_in = input_load_time + cm_latency_per
        
        partial_sum_load_time = dram_store_cycle
        time_one_iter_in_psum = max(input_load_time + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_w_psum = max(dram_read_cycle[0] + partial_sum_load_time, time_one_noc_pipe_flow)
        time_one_iter_in_k = max(dram_read_cycle[0] + input_load_time, time_one_noc_pipe_flow)

        # print(time_one_iter_w,time_one_iter_in,time_one_noc_pipe_flow)
        
        ##内层循环n2
        n2 = int(math.ceil(w_params[1]/self.config["TILE_NUM"]))

        
        cp_latency_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            cp_latency_cycle_in[i] = n2 * cp_latency_cycle[i]#每个算子每一个完整内层循环的计算时间
        # print(cp_latency_cycle_in,cp_latency_cycle,n2)

            # print(cp_latency_cycle_in[i])
            
        cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            
        dram_read_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            if i == 0:
                dram_read_cycle_in[i] = n2 * dram_read_cycle[i]#每一个完整内层循环的DRAM读取数据时间;本身已经是一个数组了
            else:
                dram_read_cycle_in[i] = dram_read_cycle[i]
            
        dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
    
    
        #外层循环n1
        n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))

        
        Each_cp_latency = [0 for _ in range(len_cp)]#每一个算子的总计算时间
        total_cp_latency = 0#总计算时间
        for i in range(len_cp):
            if i == 0:
                Each_cp_latency[i] = n1 * cp_latency_cycle_in[i] * split_k
            else:
                Each_cp_latency[i] = n1 * cp_latency_cycle_in[i]
            total_cp_latency = total_cp_latency + Each_cp_latency[i]
        # print(Each_cp_latency,n1,split_k)
            
        total_cm_latency = n1 * cm_latency_cycle_in * split_k#总通信时间
            
        Each_dram_read = [0 for _ in range(len_cp)]#每个算子的DRAM读取时间
        total_dram_read = 0#总DRAM读取时间
        for i in range(len_cp):
            Each_dram_read[i] = n1 * dram_read_cycle_in[i] * split_k 
            total_dram_read = total_dram_read + Each_dram_read[i]
        # print(Each_dram_read)
            
        total_dram_store = n1 * dram_store_cycle_in#总DRAM存储时间
        
        total_DRAM = total_dram_read + total_dram_store + n1 * input_load_time * split_k
        # print(total_dram_read,total_dram_store,input_load_time,n1*input_load_time)
        
        initial_load_time = input_load_time + dram_read_cycle[0]
        iter_over_weight_time = (n2 - 1) * time_one_iter_w * n1
        iter_over_input_time = (n1 - 1) * time_one_iter_in

        iter_over_in_psum = (n1 - 1) * (split_k - 1) * time_one_iter_in_psum
        iter_over_w_psum = (n2 - 1) * n1 * (split_k - 1) * time_one_iter_w_psum
        iter_over_in_k = (split_k - 1) * time_one_iter_in_k
        last_iter_time = time_one_noc_pipe_flow
        if iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time >= total_cp_latency:
            latency = initial_load_time + iter_over_weight_time + iter_over_input_time + iter_over_in_psum + iter_over_w_psum + iter_over_in_k + last_iter_time + dram_store_per
        else:
            latency = initial_load_time + total_cp_latency + dram_store_per
        # latency = initial_load_time + iter_over_weight_time + iter_over_input_time + last_iter_time + dram_store_per
        # print(initial_load_time,iter_over_weight_time, iter_over_input_time, iter_over_in_psum, iter_over_w_psum, iter_over_in_k)
        # print(n2,n1,time_one_iter_w)
        
        Utilization = total_cp_latency / latency
        
        return total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization

    def FlashAttention(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops, verification_flag, details):
        # i_params = [MBytes([dims[0], tx, dims[2]])+Q_RoPE_wsize, head*math.ceil(dims[1]//tx)]  # #head个Embedding矩阵和Q同时载入
        # o_params = [MBytes([dims[0], tx, dims[2]]),head*math.ceil(dims[1]//tx)]  # Question： 没有考虑mask？
        # w_params = [2*MBytes([dims[0], ty, dims[2]]) +K_RoPE_wsize, math.ceil(dims[1]//ty)]  # K+V
        # vector_cp_size = model.config['B']*tx*model.config['H_A']//model.config['N_A'] +  model.config['B']*ty * model.config['H_A']//model.config['N_A']  # RoPE
        # flash_vector_cp_size =model.config['B']* 5*tx*ty  # *dims[2]
        # cp = [[vector_cp_size/G, 0], [model.config['B']*2*2*tx*ty*dims[2]/G, 1],[flash_vector_cp_size/G, 0]]
        # LOAD: Ebd[Br, H_A]
        # each outter loop
        # LOAD: K[B,Bc,#head, H_A], V[B,Bc,#head, H_A]
        # each inner loop
        # LOAD: Q[B,Br,#head, H_A]  NOTE: K,V are inputs, NoC comm Q, Ebd is const
        # LOAD: l_old[Br,1], m_old[Br, 1], MASK[Br,Bc]
        # Comp RoPE: Vector OP 2*mul+1*sub+1*add(H_A/2 for each)-->2*Br*H_A, Q&K both need
        # Comp RMSNorm: Q&K both need. Comm Q at the same time
        # Comp Attn Score: S_ij=Q@K.T GEMM OP: 2*Br*Bc
        # Comp Attn Mask: Vector OP 1*add-->Br*Bc
        # Comp local: m_ij=rowmax(S_ij) 
        #             P_ij=exp(S_ij-m_ij) Vector OP: 1*sub+1*exp-->2*Br*Bc, 
        #             l_ij=rowsum(P_ij) vector OP: 1*add-->Br*Bc
        # Comp global: m_new=max(m_old, m_ij), 
        #              l_new=exp(m_old-m_new)*l_old + exp(m_ij-l_ij)*l_ij Vector OP: 2*mul+1*add-->3*Br
        # Dropout
        # Comp O: O_i = diag(1/l_new)*diag
        local_last = 1
        local_next = 0
        len_cp = len(cp)
        
        if i_params[1] > w_params[1]:  # input切的份数比weight多 权重驻留的情况
            self.swap_values(i_params, w_params)
        
        DDR_time = 0.001 * self.config["DRAM_LATENCY(us)"]
        # CP[1]=Q@K.T@V 每个浮点运算次数2*tx*ty*H_A
        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        cp_latency_per = [0 for _ in range(len_cp)]
        for i in range(len_cp):  # XXX: Embedding矩阵会进行 2次mul, 1次sub, 1次add(每个H_A/2)维度
            cp_latency_per[i] = self.Computation_latency(cp[i])#每一个算子每一次的计算时间,由数组表示
        # print(cp_latency_per)
        # XXX: Embedding矩阵都是一样的为什么要通信
        cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
        cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_store_per = DDR_time + self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        # XXX: 每个内循环会load in Br*1的l_i和m_i
        input_load_time = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]

    
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        #一轮的总用时latency_cycle
        cp_latency_cycle = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            cpi = cp[i]
            if cpi[1] == 0:
                w_cm_flag = 0
            else:
                w_cm_flag = self.config["TILE_NUM"] - 1
            cp_latency_cycle[i] = (w_cm_flag + 1) * cp_latency_per[i]#每一轮的计算时间
            # print(cp_latency_cycle[i])
            
        w_cm_flag = self.config["TILE_NUM"] - 1
        cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间；len_cp不为1时也只有一个算子（GEMM）进行通信
            
        dram_read_cycle = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间；本身已经是一个数组了
        for i in range(len_cp):
            dram_read_cycle[i] = dram_read_cycle[i]

        dram_store_cycle = dram_store_per#每一轮DRAM存储数据的时间
        
        
        ##内层循环n2
        n2 = int(math.ceil(w_params[1]/self.config["TILE_NUM"]))

        #print(n2)

        
        cp_latency_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            if i == 1:
                cp_latency_cycle_in[i] = n2 * cp_latency_cycle[i]#每个算子每一个完整内层循环的计算时间
            else:
                cp_latency_cycle_in[i] = cp_latency_cycle[i]
            # print(cp_latency_cycle_in[i])
            
        cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            
        dram_read_cycle_in = [0 for _ in range(len_cp)]
        for i in range(len_cp):
            if i == 1:
                dram_read_cycle_in[i] = n2 * dram_read_cycle[i]#每一个完整内层循环的DRAM读取数据时间;本身已经是一个数组了
            else:
                dram_read_cycle_in[i] = dram_read_cycle[i]
            
        dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
        
        input_cycle_in = input_load_time + dram_read_cycle[1]
        cp_cycle_in = cp_latency_cycle_in[0] + cp_latency_cycle_in[1] + cp_latency_cycle_in[2]
        latency_cycle_in = max(input_cycle_in,cp_cycle_in)
        #外层循环n1
        n1 = int(math.ceil(i_params[1]/self.config["TILE_NUM"]))


        
        Each_cp_latency = [0 for _ in range(len_cp)]#每一个算子的总计算时间
        total_cp_latency = 0#总计算时间
        for i in range(len_cp):
            Each_cp_latency[i] = n1 * cp_latency_cycle_in[i]
            total_cp_latency = total_cp_latency + Each_cp_latency[i]
            # print(Each_cp_latency[i],n1)
            
        total_cm_latency = n1 * cm_latency_cycle_in#总通信时间
            
        Each_dram_read = [0 for _ in range(len_cp)]#每个算子的DRAM读取时间
        total_dram_read = 0#总DRAM读取时间
        for i in range(len_cp):
            Each_dram_read[i] = n1 * dram_read_cycle_in[i]
            total_dram_read = total_dram_read + Each_dram_read[i]
            
        total_dram_store = n1 * dram_store_cycle_in#总DRAM存储时间
        
        total_DRAM = total_dram_read + total_dram_store + n1 * input_load_time
        
        
        latency = input_cycle_in + cp_cycle_in - cp_latency_cycle_in[0] + (n1 - 1) * latency_cycle_in + dram_store_cycle
        #print(dram_store_cycle_in,n1,input_load_time)
        if latency <= total_cp_latency:
            latency = total_cp_latency + dram_store_cycle
        
        Utilization = total_cp_latency / latency
        
        return total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization 
    
    def swap_values(self, i_params, w_params):
        if i_params[1] > w_params[1]:
        # 交换 i_params 和 w_params 的值
            i_params, w_params = w_params, i_params

    # def Computation_latency(self, cp_n):
    #     if cp_n[1] == 0:
    #         cp_latency = cp_n[0] / self.config["VECTOR(TOPS)"]#cp_type=0为Vector计算
    #     else:
    #         cp_latency = cp_n[0] / self.config["GEMM(TFLOPS)"]#cp_type=1为GEMM计算
    #     return cp_latency

    def Computation_latency(self, cp_n):
        """Calculates computation latency based on operation type.

        Args:
            cp_n: A list or tuple [cp_size, cp_type] where cp_size is the computational cost (GFLOPs) and cp_type is the operation type (0: Vector, 1: GEMM).

        Returns:
                The computation latency in seconds.  Returns 0 if cp_type is invalid.
            """
        cp_size, cp_type = cp_n
        latency_lookup = {
            0: self.config["VECTOR(TOPS)"],
            1: self.config["GEMM(TFLOPS)"],
        }
        performance = latency_lookup.get(cp_type, 0) #Returns 0 if cp_type is not 0 or 1.
        return cp_size / performance if performance else 0

    def Mode(self, i_params, o_params, w_params, cp):
        len_cp = len(cp)
        
        if len_cp == 1:#一个算子
            cp0 = cp[0]
            if cp0[1] == 0:
                mode = 10#一个RMSNorm算子/Vector算子
            else:
                mode = 11#一个GEMM算子
        elif len_cp == 2: #两个算子
            cp0 = cp[0]
            cp1 = cp[1]
            if cp0[1] == 0 and cp1[1] == 1:#先Vector，后GEMM
                mode = 21
            else:#先GEMM，后Vector
                mode = 22
        else:#len_cp = 3,flash-attention
            mode = 31
        
        return mode

    def Communication_latency(self, cm_size, cm_type, cm_hops):
        cm_latency = cm_size / self.config["NOC_BW(GB/s)"]#通信延迟，单位为ms
        overlap = self.config["HOP_LATENCY(us)"] * cm_hops#overlap用时，单位为us
        return cm_latency, overlap
    
    def DRAM_read(self, i_params, w_params, local_last, cp, len_cp):        
        #print(i_params, w_params)
        # 为每个tile读取一份被切分后的
        if len_cp == 1:  # 单个算子只计算weight的读取时间，默认input在本地
            dram_size = i_params[0] + w_params[0]
            read_latency = (dram_size - i_params[0]) * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]

        elif len_cp == 2:
            read_latency = [0 for _ in range(len_cp)]
            for i in range(len_cp):  # 两个算子
                cpi = cp[i]
                if cpi[1] == 0:  # 算子是vector类型
                    dram_size = 0
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] 
                else:  # 算子是GEMM类型，只用读取weight (疑问 为什么连续读两个weight需要加上DRAM_LATENCY)
                    dram_size = w_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]
        else:  # 需要3个及以上的vector/matrix进行计算的算子
            read_latency = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                if i == 0:
                    dram_size = 0
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] 
                elif i == 1:
                    dram_size = w_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] + 0.001 * self.config["DRAM_LATENCY(us)"]
                else:
                    read_latency[i] == 0
        #print(read_latency)      
        return read_latency
    
    def DRAM_store(self, i_params, o_params, w_params, local_next):
        if i_params[1] <= self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"]:
            o_flag = i_params[1]  # vector situation
        else:
            o_flag = i_params[1] * w_params[1]  # GEMM situation
        
        if local_next == 1 and o_flag == 16: # 数据不用传输且刚好输出切分16份(可以均匀地分配到tile上)时，输出复用为输入
            store_latency = 0
        else:  # 16个tile上的SRAM存回DRAM
            store_latency = (o_params[0] * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"]
        return store_latency


if __name__ == "__main__":
    tx8_config = load_config("./hardware_parameter.json")
    arch = Tx8(tx8_config)

    #使用Projection模块第一个RMSNorm算子验证其结果是否正确
    print("GEMM:")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([0.5, 1, 64], [96.0, 1], [1.5, 1, 64], [[6.442450944, 1]], 1.5, 0, 5, False)
    print("是否满足SRAM要求:", verification_result_0)
    print("总计算时间:", total_cp_latency_0)
    print("总通信时间:", total_cm_latency_0)
    print("总访存时间:", total_DRAM_0)
    print("总延迟:", latency_0)
    print("利用率:", Utilization_0)

    # #使用Projection模块第一个RMSNorm算子验证其结果是否正确
    # print("Projection的RMSNorm算子验证(后续不复用):")
    # verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([2,16,1], [2,16,1],[0.00048828,16], [[0.00390625,0]], 0, 0,  0,False)
    # print("是否满足SRAM要求:", verification_result_0)
    # print("总计算时间:", total_cp_latency_0)
    # print("总通信时间:", total_cm_latency_0)
    # print("总访存时间:", total_DRAM_0)
    # print("总延迟:", latency_0)
    # print("利用率:", Utilization_0)
    
    # # 使用Projection模块第一个GEMM算子验证其结果是否正确
    # print("GEMM算子验证(tile_m = 256, tile_n = 32, split_k = 1):")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2,16,1], [0.015625,2048],[0.25,128,1], [[0.0625,1]], 0.25, 0, 5, False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)
    
    # # 使用Projection模块第一个GEMM算子验证其结果是否正确
    # print("GEMM算子验证(tile_m = 256, tile_n = 32, split_k = 2):")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([1,16,2], [0.015625,2048],[0.125,128,2], [[0.03125,1]], 0.125, 0, 5,False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)
    
    
    
    # #FFN检验
    # print("FFN检验(2.885681152):")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.25, 128,1], [0.02099609375, 4096], [2.6875, 32,1], [[0.090177536, 1]], 2.6875, 0, 5,False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)
    
    # #FFN与SiLU融合检验
    # print("FFN与SiLU融合检验:")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.0625, 512,1], [0.000244140625, 352256], [0.125, 688,1], [[0.001048576, 1], [5.12e-07, 0]], 0.125, 0, 5,False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)

    # #FFN与SiLU融合检验
    # print("FFN与SiLU融合检验:")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.009765625, 16, 4096], [0.341796875, 256], [6.67572021484375e-05, 16, 4096], [[0.00035264, 1], [0.00070528, 0]], 6.67572021484375e-05, 0, 5, False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)
    
    # #单FFN检验
    # print("单FFN检验:")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.0625, 512,1], [0.000244140625, 352256], [0.125, 688,1], [[0.001048576, 1]], 0.125, 0, 5,False)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # print("总通信时间:", total_cm_latency)
    # print("总访存时间:", total_DRAM)
    # print("总延迟:", latency)
    # print("利用率:", Utilization)
    
    # #单SiLU检验
    # print("单SiLU检验:")
    # verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.0625, 352256], [0.000244140625, 352256], [0.125, 688], [[5.12e-07, 0]], 0.125, 0, 5)
    # print("是否满足SRAM要求:", verification_result)
    # print("总计算时间:", total_cp_latency)
    # # print("总通信时间:", total_cm_latency)
    # # print("总访存时间:", total_DRAM)
    # # print("总延迟:", latency)
    # # print("利用率:", Utilization)
    
    

# # 创建Tx8类的实例
# arch = Tx8()
    '''
    #使用Projection模块第一个RMSNorm算子验证其结果是否正确
    print("Projection的RMSNorm算子验证(后续不复用):")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([2,16,1], [2,16,1],[0.00048828,16], [[0.00390625,0]], 0, 0,  0,False)
    print("是否满足SRAM要求:", verification_result_0)
    print("总计算时间:", total_cp_latency_0)
    print("总通信时间:", total_cm_latency_0)
    print("总访存时间:", total_DRAM_0)
    print("总延迟:", latency_0)
    print("利用率:", Utilization_0)
    
    # 使用Projection模块第一个RMSNorm算子验证其结果是否正确/不同切分次数
    print("Projection的RMSNorm算子验证(后续不复用), 切成32份:")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([1,32], [1,32],[0.00024414,32], [[0.001953125,0]], 0, 0,  0)
    print("是否满足SRAM要求:", verification_result_0)
    print("总计算时间:", total_cp_latency_0)
    print("总通信时间:", total_cm_latency_0)
    print("总访存时间:", total_DRAM_0)
    print("总延迟:", latency_0)
    print("利用率:", Utilization_0)


    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证(tile_m = 256, tile_n = 32):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2,16], [0.015625,2048],[0.25,128], [[0.0625,1]], 0.25, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证(tile_m = 128, tile_n = 32):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([1,32], [0.0078125,4096],[0.25,128], [[0.03125,1]], 0.125, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证(tile_m = 32, tile_n = 128):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.25,128], [0.0078125,4096],[1,32], [[0.03125,1]], 0.125, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证(tile_m = 64, tile_n = 64):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.5,64], [0.0078125,4096],[0.5,64], [[0.03125,1]], 0.125, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证(tile_m = 128, tile_n = 64):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([1,32], [0.015625,2048],[0.5,64], [[0.0625,1]], 0.25, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    
    # Projection模块的RMSNorm算子与一个GEMM算子融合并验证其结果是否正确
    print("Projection的RMSNorm与GEMM算子融合验证:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2.00048828,16], [0.015625,2048],[0.25,128], [[0.00390625,0],[0.0625,1]], 0.25, 0,  5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    
    # 非法输入测试
    print("Projection的GEMM算子验证(非法数据):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([32.0, 1] ,[86.0, 1],[86.0, 1] ,[[344.0, 1], [0.16796875, 0]] ,86.0 ,0 ,5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # 极端输入情况测试
    print("Projection的GEMM算子验证(极端输入情况):")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.5078125, 64],[0.0001220703125, 1048576],[0.0078125, 16384],[[0.0009765625, 0], [0.0003662109375, 1]],0.0078125, 0, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    '''
    # flash-attention测试
    print("flash-attention算子验证1:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.0625, 640], [0.03125, 640], [0.09375, 320], [[0.00065536, 0], [0.016777216, 1], [0.00032768, 0]], 0.09375, 0, 1,False)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    '''
    # flash-attention测试
    print("flash-attention算子验证2:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.125, 512], [0.125, 512], [0.1875, 16], [[6.103515625e-05, 0], [0.03125, 1], [0.00030517578125, 0]], 0.1875, 0, 1 )
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # flash-attention测试
    print("flash-attention算子验证3:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2.0, 1], [2.0, 1], [0.1875, 16], [[0.000518798828125, 0], [0.5, 1], [0.0048828125, 0]], 0.1875, 0, 1 )
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)

    print("flash-attention算子验证4:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([0.00390625, 512] ,[0.0001220703125, 512] ,[0.005859375, 512] ,[[0, 0], [3.2768e-05, 1], [0, 0]] ,0.005859375, 0 ,1 )
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)  
    '''