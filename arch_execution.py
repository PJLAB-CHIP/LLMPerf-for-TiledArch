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


class Tx8:
    '''
    input: config (TILE_NUM, sram_size, noc_bandwidth, DRAM_BW(GB/s), Gemm, VECTOR(TOPS), One_Hops) == (tiles, sram, noc, dram, gemm, vector, Hops)
    '''
    
    def __init__(self, config) -> None:
        self.config = config
    def execute(self, i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops):
        # i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
        # o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等; 对于GEMM，o_flag = i_flag * w_flag; 对于FLashAttention，确保i_flag = o_flag必须要满足这两种情况
        # w_params = [w_size, w_flag] w_size为一份输出的大小，单位为MB；w_flag为输出的总份数，例如16/32/64等
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
        
        
        verification_result = self.verification(i_params, o_params, w_params)#要求(1)是否可以计算;True or False
        
        
        #计算local_last和local_next；Dram_read和Dram_store函数的参数
        len_cp = len(cp)
        cp0 = cp[0]
        if len_cp == 1:
            if cp0[1] == 1 and i_params[1] <= self.config["TILE_NUM"]:
                local_last = 0
                local_next = 0
            else:
                local_last = 0
                local_next = 0
        else:
            local_last = 1
            local_next = 0    


        #一次的计算时间、通信时间和存储时间；cp_latency_per, cm_latency_per, dram_store_per
        if len_cp == 1:#仅有一个算子时
            cp_latency_per = self.Computation_latency(cp[0])#每一次的计算时间，单位ms
            
            cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
            cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms
            
            dram_store_per = self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms

        else:#RMSNorm+GEMM（len_cp == 2)和flash-attention, vector-GEMM-vector连接（len_cp == 3)
            cp_latency_per = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                cp_latency_per[i] = self.Computation_latency(cp[i])#每一个算子每一次的计算时间,由数组表示

            cm_latency_per_noc, cm_latency_per_overlap = self.Communication_latency(cm_size, cm_type, cm_hops)
            cm_latency_per = cm_latency_per_noc + 0.001 * cm_latency_per_overlap#每一次的通信时间，单位为ms；len_cp不为1时也只有一个算子（GEMM）进行通信
            
            dram_store_per = self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间，单位为ms
        
        
        
        #一轮的计算时间、通信时间、读取时间和存储时间；cp_latency_cycle, cm_latency_cycle, dram_read_cycle, dram_store_cycle; 核心指标为通信次数w_cm_flag
        if len_cp == 1:#仅有一个算子时
            if i_params[1] == o_params[1] == w_params[1]:
                w_cm_flag = 0
            else:
                w_cm_flag = self.config["TILE_NUM"] - 1#通信次数w_cm_flag的计算
            
            cp_latency_cycle = (w_cm_flag + 1) * cp_latency_per#每一轮的计算时间
            cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间
            dram_read_cycle = 0.001 * self.config["DRAM_LATENCY(us)"] + self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间,这部分只含权重
            dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
            #print(cp_latency_cycle,cp_latency_cycle,dram_store_cycle)
            #print(dram_read_cycle)
            
        else:#RMSNorm+GEMM（len_cp == 2)和flash-attention, vector-GEMM-vector连接（len_cp == 3)
            cp_latency_cycle = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                cpi = cp[i]
                if cpi[1] == 0:
                    w_cm_flag = 0
                else:
                    w_cm_flag = self.config["TILE_NUM"] - 1
                cp_latency_cycle[i] = (w_cm_flag + 1) * cp_latency_per[i]#每一轮的计算时间
                #print(cp_latency_cycle[i])
            
            cm_latency_cycle = w_cm_flag * cm_latency_per#每一轮的通信时间；len_cp不为1时也只有一个算子（GEMM）进行通信
            
            dram_read_cycle = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间；本身已经是一个数组了
            for i in range(len_cp):
                dram_read_cycle[i] = dram_read_cycle[i] + 0.001 * self.config["DRAM_LATENCY(us)"]
            dram_store_cycle = (w_cm_flag + 1) * dram_store_per#每一轮DRAM存储数据的时间
            
            #print(dram_read_cycle)
        
        
        
        # 内层循环n2
        if i_params[1] == o_params[1] == w_params[1]:
            n2 = 1
        else:
            if i_params[1] <= w_params[1]:#输入驻留
                n2 = w_params[1]/self.config["TILE_NUM"]
            else:#权重驻留
                n2 = i_params[1]/self.config["TILE_NUM"]
        
        if len_cp == 1:#仅有一个算子时
            cp_latency_cycle_in = n2 * cp_latency_cycle#每一个完整内层循环的计算时间
            cm_latency_cycle_in = n2 * cm_latency_cycle#每一个完整内层循环的通信时间
            dram_read_cycle_in = n2 * dram_read_cycle + i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]#每一个完整内层循环的DRAM读取数据时间
            dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
            #print(dram_read_cycle_in,dram_store_cycle_in)
            
        else:#RMSNorm+GEMM（len_cp == 2)和flash-attention, vector-GEMM-vector连接（len_cp == 3)
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
                if i == 2:
                    dram_read_cycle_in[i] = n2 * dram_read_cycle[i]#每一个完整内层循环的DRAM读取数据时间;本身已经是一个数组了
                else:
                    dram_read_cycle_in[i] = dram_read_cycle[i]
            
            dram_store_cycle_in = n2 * dram_store_cycle#每一个完整内层循环的DRAM存储数据时间
            #print(dram_read_cycle_in[0],dram_read_cycle_in[1])
        
        
        # 外层循环n1
        if i_params[1] <= w_params[1]:
            n1 = i_params[1]/self.config["TILE_NUM"]
        else:
            n1 = w_params[1]/self.config["TILE_NUM"]
        if len_cp == 1:#仅有一个算子时
            total_cp_latency = n1 * cp_latency_cycle_in
            total_cm_latency = n1 * cm_latency_cycle_in
            total_dram_read = n1 * dram_read_cycle_in
            total_dram_store = n1 * dram_store_cycle_in
        else:
            #RMSNorm+GEMM（len_cp == 2)和flash-attention, vector-GEMM-vector连接（len_cp == 3)
            Each_cp_latency = [0 for _ in range(len_cp)]#每一个算子的总计算时间
            total_cp_latency = 0#总计算时间
            for i in range(len_cp):
                Each_cp_latency[i] = n1 * cp_latency_cycle_in[i]
                total_cp_latency = total_cp_latency + Each_cp_latency[i]
            
            total_cm_latency = n1 * cm_latency_cycle_in#总通信时间
            
            Each_dram_read = [0 for _ in range(len_cp)]#每个算子的DRAM读取时间
            total_dram_read = 0#总DRAM读取时间
            for i in range(len_cp):
                Each_dram_read[i] = n1 * dram_read_cycle_in[i]
                total_dram_read = total_dram_read + Each_dram_read[i]
            
            total_dram_store = n1 * dram_store_cycle_in#总DRAM存储时间
            #print(total_dram_read,total_dram_store)
            
            
        #总延迟latency计算
        if len_cp == 1:#仅有一个算子时
            variables = {
                "cp_latency_cycle": cp_latency_cycle,
                "cm_latency_cycle": cm_latency_cycle,
                "dram_read": dram_read_cycle,
                "dram_store_cycle": dram_store_cycle
            }
            max_variable = max(variables, key=variables.get)#找出最大值及其对应的变量名
            dram_read_cycle_in_I = i_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]
            dram_read_cycle_in_W = w_params[0] * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]
            #print(dram_read_cycle_in_I,dram_read_cycle, cp_latency_cycle)
            if i_params[1] <= 16:
                if max_variable ==  "cp_latency_cycle":
                    latency = dram_read_cycle_in_I + dram_read_cycle + dram_store_per + total_cp_latency
                elif max_variable ==  "cm_latency_cycle":
                    latency = dram_read_cycle_in_I + dram_read_cycle + dram_store_per + total_cm_latency
                elif max_variable ==  "dram_read":
                    latency = total_dram_read + dram_store_per + cp_latency_per
                else:
                    latency = total_dram_store + dram_read_cycle + cp_latency_per + dram_read_cycle_in_I
            else:
                if i_params[1] <= w_params[1]:
                    if cp_latency_cycle < dram_read_cycle_in_I + dram_read_cycle:
                        latency = dram_read_cycle_in_I + dram_read_cycle + dram_store_per + total_cp_latency + (n1 - 1) * (dram_read_cycle_in_I + 2 * dram_read_cycle - 2 * cp_latency_cycle)
                    else:
                        latency = dram_read_cycle_in_I + dram_read_cycle + dram_store_per + total_cp_latency
                else:
                    if cp_latency_cycle < dram_read_cycle_in_W + dram_read_cycle:
                            latency = dram_read_cycle_in_W + dram_read_cycle + dram_store_per + total_cp_latency + (n1 - 1) * (dram_read_cycle_in_W + 2 * dram_read_cycle - 2 * cp_latency_cycle)
                    else:
                        latency = dram_read_cycle_in_W + dram_read_cycle + dram_store_per + total_cp_latency
                    
        elif len_cp == 2:#RMSNorm+GEMM
            variables = {
                "cp_latency_cycle": cp_latency_cycle[1],
                "cm_latency_cycle": cm_latency_cycle,
                "dram_read": dram_read_cycle[1],
                "dram_store_cycle": dram_store_cycle
            }
            max_variable = max(variables, key=variables.get)#找出最大值及其对应的变量名
            if dram_read_cycle[1] > cp_latency_cycle[0]:
                if max_variable ==  "cp_latency_cycle":
                    latency = dram_read_cycle[0] + dram_read_cycle[1] + Each_cp_latency[1] + dram_store_per
                elif max_variable ==  "cm_latency_cycle":
                    latency = dram_read_cycle[0] + dram_read_cycle[1] + dram_store_per + total_cm_latency
                elif max_variable ==  "dram_read":
                    latency = total_dram_read + dram_store_per + cp_latency_per[1]
                else:
                    latency = total_dram_store + dram_read_cycle + cp_latency_per[1]
            else:
                if max_variable ==  "cp_latency_cycle":
                    latency = dram_read_cycle[0] + total_cp_latency + dram_store_per
                elif max_variable ==  "cm_latency_cycle":
                    latency = dram_read_cycle[0] + Each_cp_latency[0] + dram_store_per + total_cm_latency
                elif max_variable ==  "dram_read":
                    latency = total_dram_read + dram_store_per + cp_latency_per[1]
                else:
                    latency = total_dram_store + dram_read_cycle + cp_latency_per[1]
        else:#flash-attention, vector-GEMM-vector连接（len_cp == 3)
            variables = {
                "cp_latency_cycle": cp_latency_cycle[1],
                "cm_latency_cycle": cm_latency_cycle,
                "dram_read": dram_read_cycle[1],
                "dram_store_cycle": dram_store_cycle
            }
            max_variable = max(variables, key=variables.get)#找出最大值及其对应的变量名
            if dram_read_cycle[1] > cp_latency_cycle[0]:
                if max_variable ==  "cp_latency_cycle":
                    latency = dram_read_cycle[0] + dram_read_cycle[1] + Each_cp_latency[1] + dram_store_cycle_in
                elif max_variable ==  "cm_latency_cycle":
                    latency = dram_read_cycle[0] + dram_read_cycle[1] + dram_store_cycle_in + total_cm_latency
                elif max_variable ==  "dram_read":
                    latency = total_dram_read + dram_store_cycle_in + cp_latency_per[1]
                else:
                    latency = total_dram_store + dram_read_cycle + cp_latency_per[1]
            else:
                if max_variable ==  "cp_latency_cycle":
                    latency = dram_read_cycle[0] + total_cp_latency + dram_store_cycle_in
                elif max_variable ==  "cm_latency_cycle":
                    latency = dram_read_cycle[0] + Each_cp_latency[0] + dram_store_cycle_in + total_cm_latency
                elif max_variable ==  "dram_read":
                    latency = total_dram_read + dram_store_cycle_in + cp_latency_per[1]
                else:
                    latency = total_dram_store + dram_read_cycle + cp_latency_per[1]
            
            
            
        #利用率Utilization计算
        Utilization = total_cp_latency / latency

        #print('output',verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization)
        total_DRAM = total_dram_read + total_dram_store#总DRAM访存时间
        return verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
    
    def verification(self, i_params, o_params, w_params):
    # 该函数用于验证SRAM是否够用。如果切分数量小于等于16，则不需要遵循SRAM的2-3-2分配策略（2输入，3权重，2输出） 
        o_flag = o_params[1]
        
        if i_params[1] == o_flag == w_params[1]:#Vector算子情况，如RMSNorm
            if i_params[1] <= self.config["TILE_NUM"]:
                if i_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False
            else:
                if 2 * i_params[0] + 2 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

        else:
            if i_params[1] and o_flag <= self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"] and i_params[1] == o_flag:
                if i_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False
        
            elif i_params[1] and o_flag <= self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"] and i_params[1] != o_flag:
                if i_params[0] + o_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False
                
            elif i_params[1] <= self.config["TILE_NUM"] and o_flag <= self.config["TILE_NUM"] and w_params[1] > self.config["TILE_NUM"]:
                if i_params[0] + o_params[0] + 3 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

            elif i_params[1] <= self.config["TILE_NUM"] and o_flag > self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"]:
                if i_params[0] + 2 * o_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

            elif i_params[1] <= self.config["TILE_NUM"] and o_flag > self.config["TILE_NUM"] and w_params[1] > self.config["TILE_NUM"]:
                if i_params[0] + 2 * o_params[0] + 3 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

            elif i_params[1] > self.config["TILE_NUM"] and o_flag <= self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"]:
                if 2 * i_params[0] + o_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False
 
            elif i_params[1] > self.config["TILE_NUM"] and o_flag <= self.config["TILE_NUM"] and w_params[1] > self.config["TILE_NUM"]:
                if 2 * i_params[0] + o_params[0] + 3 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

            elif i_params[1] > self.config["TILE_NUM"] and o_flag > self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"]:
                if 2 * i_params[0] + 2 * o_params[0] + w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

            else:
            # 情况8：i_flag > 16 and o_flag > 16 and w_flag > 16，这也是最常见的情况
                if 2 * i_params[0] + 2 * o_params[0] + 3 * w_params[0] <= self.config["SRAM(MB)"]:
                    verification_result = True
                else:
                    verification_result = False

        return verification_result


    def Computation_latency(self, cp_n):
        if cp_n[1] == 0:
            cp_latency = cp_n[0] / self.config["VECTOR(TOPS)"]#cp_type=0为Vector计算
        else:
            cp_latency = cp_n[0] / self.config["GEMM(TFLOPS)"]#cp_type=1为GEMM计算
        return cp_latency


    def Communication_latency(self, cm_size, cm_type, cm_hops):
        cm_latency = cm_size / self.config["NOC_BW(GB/s)"]#通信延迟，单位为ms
        overlap = self.config["HOP_LATENCY(us)"] * cm_hops#overlap用时，单位为us
        return cm_latency, overlap
    
    def DRAM_read(self, i_params, w_params, local_last, cp, len_cp):        
        if len_cp == 1:
            dram_size = i_params[0] + w_params[0]
            if i_params[1] <= w_params[1]:
                read_latency = (dram_size - i_params[0]) * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]
            else:
                read_latency = (dram_size - w_params[0]) * self.config["TILE_NUM"] / self.config["DRAM_BW(GB/s)"]
        elif len_cp == 2:
            read_latency = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                cpi = cp[i]
                if cpi[1] == 0:
                    dram_size = i_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] 
                else:
                    dram_size = w_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"]
        else:
            read_latency = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                if i == 0:
                    dram_size = i_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"] 
                elif i == 1:
                    dram_size = w_params[0]
                    read_latency[i] = (dram_size * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"]
                else:
                    read_latency[i] == 0
        #print(read_latency)      
        return read_latency
    
    def DRAM_store(self, i_params, o_params, w_params, local_next):
        if i_params[1] <= self.config["TILE_NUM"] and w_params[1] <= self.config["TILE_NUM"]:
            o_flag = i_params[1]
        else:
            o_flag = i_params[1] * w_params[1]#用于计算输出的切分次数
        
        if local_next == 1 and o_flag == 16: #输出复用为下一个算子的输入
            store_latency = 0
        else:#不复用
            store_latency = (o_params[0] * self.config["TILE_NUM"]) / self.config["DRAM_BW(GB/s)"]
        return store_latency


if __name__ == "__main__":
    tx8_config = load_config("./hardware_parameter.json")
    arch = Tx8(tx8_config)

# # 创建Tx8类的实例
# arch = Tx8()

    # 使用Projection模块第一个RMSNorm算子验证其结果是否正确
    print("Projection的RMSNorm算子验证(后续不复用):")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([2,16], [2,16],[0.00048828,16], [[0.00390625,0]], 0, 0,  0)
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
    
    # Projection模块的RMSNorm算子与一个GEMM算子融合并验证其结果是否正确
    print("Projection的RMSNorm与GEMM算子融合验证:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2.00048828,16], [0.015625,2048],[0.25,128], [[0.00390625,0],[0.0625,1]], 0.25, 0,  5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    
    
