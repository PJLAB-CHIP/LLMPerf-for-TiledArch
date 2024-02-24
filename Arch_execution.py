import json
#import tbk

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
    input: config (4X4 tile, sram_size, noc_bandwidth, Dram_bandwidth, Gemm, Vector, One_Hops) == (tiles, sram, noc, dram, gemm, vector, Hops)
    '''
    
    def __init__(self, config) -> None:
        self.config = config
    def execute(self, i_params, o_params, w_params, cp, cm_size, cm_type, local_last, local_next, cm_hops):
        # i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
        # o_params = [o_size] o_size为一份输出的大小，单位为MB
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
        # o_flag为输出的总份数，o_flag = o_params[1] = i_params[1] * w_params[1]
        
        # 更改的部分
        # cp = [cp0, cp1]，其中cp0 = [cp_size0, cp_type0]
        
                
        len_cp = len(cp)
        cp0 = cp[0]
        if len_cp == 1 and cp0[1] == 1:
            local_last = 1
            local_next = 0
        elif len_cp == 1 and cp0[1] == 0:
            local_last = 0
            local_next = 0
        else:
            local_last = 1
            local_next = 0
        
        verification_result = self.verification(i_params, o_params, w_params)#要求(1)是否可以计算
        
        Each_cp_latency = [0 for _ in range(len_cp)]
        total_cp_latency = 0
        for i in range(len_cp):
            cpi = cp[i]
            if cpi[1] == 0:
                w_cm_flag = 0
                n = 1
            else:
                n = (i_params[1] / self.config["4X4 tile"]) * (w_params[1] / self.config["4X4 tile"])
                w_cm_flag = self.config["4X4 tile"] - 1                
                
            cp_latency = self.Computation_latency(cp[i])#每一次的计算时间
            cp_latency_cycle = (w_cm_flag + 1) * cp_latency#每一轮的计算时间
            Each_cp_latency[i] = n * cp_latency_cycle#每个算子的计算时间
            total_cp_latency = total_cp_latency + Each_cp_latency[i]#总计算时间
        
        n = (i_params[1] / self.config["4X4 tile"]) * (w_params[1] / self.config["4X4 tile"])#计算出流水轮次n
            
        if w_params[1] > self.config["4X4 tile"]:
            w_cm_flag = self.config["4X4 tile"] - 1
        elif w_params[1] <= self.config["4X4 tile"] and cm_size == 0:
            w_cm_flag = 0
        else:
            w_cm_flag = w_params[1] - 1#计算出一轮内通信的次数，例如15次，则计算和存储是16次
        
        
        cm_latency_noc, overlap = self.Communication_latency(cm_size, cm_type, cm_hops)#每一次noc通信时间和overlap时间
        cm_latency = cm_latency_noc + 0.001 * overlap#每一次的通信时间，单位为ms
        cm_latency_cycle = w_cm_flag * cm_latency#每一轮的通信时间
        total_cm_latency = n * cm_latency_cycle#总通信时间
        
        if len_cp == 1:
            dram_read = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)#每一轮DRAM读取数据的时间
            total_dram_read = n * dram_read#总DRAM读取时间
            
            dram_store = self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间
            dram_store_cycle = (w_cm_flag + 1) * dram_store#每一轮DRAM存储数据的时间
            total_dram_store = n * dram_store_cycle#总DRAM存储数据时间
            total_DRAM = total_dram_read + total_dram_store#总DRAM访存时间
            
            #找到cp_latency_cycle，cm_latency_cycle，dram_read，dram_store_cycle中的最大值, max_variable存储其最大值的名称
            # 使用字典存储变量及其值
            variables = {
                "cp_latency_cycle": cp_latency_cycle,
                "cm_latency_cycle": cm_latency_cycle,
                "dram_read": dram_read,
                "dram_store_cycle": dram_store_cycle
            }

            # 找出最大值及其对应的变量名
            max_variable = max(variables, key=variables.get)
        
            # 计算不同情况下的总延迟
            if max_variable ==  "cp_latency_cycle":
                latency = dram_read + dram_store + total_cp_latency
            elif max_variable ==  "cm_latency_cycle":
                latency = dram_read + dram_store + total_cm_latency
            elif max_variable ==  "dram_read":
                latency = total_dram_read + dram_store + cp_latency
            else:
                latency = total_dram_store + dram_read + cp_latency
        
            Utilization = total_cp_latency / latency
            
        else:
            total_dram_read = 0
            dram_read_list = self.DRAM_read(i_params, w_params, local_last, cp, len_cp)
            for i in range(len_cp):
                dram_read = dram_read_list[i]
                cpi = cp[i]
                if cpi[1] == 0:
                    n = 1
                else:
                    n = (i_params[1] / self.config["4X4 tile"]) * (w_params[1] / self.config["4X4 tile"])
                Each_dram_read = n * dram_read
                total_dram_read = total_dram_read + Each_dram_read
        
            dram_store = self.DRAM_store(i_params, o_params, w_params, local_next)#每一次DRAM存储数据的时间
            dram_store_cycle = (w_cm_flag + 1) * dram_store#每一轮DRAM存储数据的时间
            total_dram_store = n * dram_store_cycle#总DRAM存储数据时间
            total_DRAM = total_dram_read + total_dram_store#总DRAM访存时间
        
            latency = dram_read_list[0] + total_cp_latency + dram_store        
        
            Utilization = total_cp_latency / latency        
        
        return verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
    
    def verification(self, i_params, o_params, w_params):
    # 该函数用于验证SRAM是否够用。如果切分数量小于等于16，则不需要遵循SRAM的2-3-2分配策略（2输入，3权重，2输出）
        
        if i_params[1] <= self.config["4X4 tile"] and w_params[1] <= self.config["4X4 tile"]:
            o_flag = i_params[1]
        else:
            o_flag = i_params[1] * w_params[1]#用于计算输出的切分次数    
    
        if i_params[1] and o_flag <= 16 and w_params[1] <= 16 and i_params[1] == o_flag:
            if i_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
        
        elif i_params[1] and o_flag <= 16 and w_params[1] <= 16 and i_params[1] != o_flag:
            if i_params[0] + o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
                
        elif i_params[1] <= 16 and o_flag <= 16 and w_params[1] > 16:
            if i_params[0] + o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] <= 16 and o_flag > 16 and w_params[1] <= 16:
            if i_params[0] + 2 * o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] <= 16 and o_flag > 16 and w_params[1] > 16:
            if i_params[0] + 2 * o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] > 16 and o_flag <= 16 and w_params[1] <= 16:
            if 2 * i_params[0] + o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
 
        elif i_params[1] > 16 and o_flag <= 16 and w_params[1] > 16:
            if 2 * i_params[0] + o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] > 16 and o_flag > 16 and w_params[1] <= 16:
            if 2 * i_params[0] + 2 * o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        else:
        # 情况8：i_flag > 16 and o_flag > 16 and w_flag > 16，这也是最常见的情况
            if 2 * i_params[0] + 2 * o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        return verification_result


    def Computation_latency(self, cp_each):
        if cp_each[1] == 0:
            cp_latency = cp_each[0] / self.config["Vector"]#cp_type=0为Vector计算
        else:
            cp_latency = cp_each[0] / self.config["Gemm"]#cp_type=1为GEMM计算
        return cp_latency


    def Communication_latency(self, cm_size, cm_type, cm_hops):
        cm_latency = cm_size / self.config["noc_bandwidth"]#通信延迟，单位为ms
        overlap = self.config["One_Hops"] * cm_hops#overlap用时，单位为us
        return cm_latency, overlap
    
    def DRAM_read(self, i_params, w_params, local_last, cp, len_cp):        
        if len_cp == 1:
            dram_size = i_params[0] + w_params[0]
            if local_last == 0:#不复用
                read_latency = (dram_size * self.config["4X4 tile"]) / self.config["Dram_bandwidth"]
            else:#复用
                read_latency = (dram_size - i_params[0]) * self.config["4X4 tile"] / self.config["Dram_bandwidth"]
        else:
            read_latency = [0 for _ in range(len_cp)]
            for i in range(len_cp):
                cpi = cp[i]
                if cpi[1] == 0:
                    dram_size = i_params[0]
                    read_latency[i] = (dram_size * self.config["4X4 tile"]) / self.config["Dram_bandwidth"] 
                else:
                    dram_size = w_params[0]
                    read_latency[i] = (dram_size * self.config["4X4 tile"]) / self.config["Dram_bandwidth"]
        #print(read_latency)      
        return read_latency
    
    def DRAM_store(self, i_params, o_params, w_params, local_next):
        if i_params[1] <= self.config["4X4 tile"] and w_params[1] <= self.config["4X4 tile"]:
            o_flag = i_params[1]
        else:
            o_flag = i_params[1] * w_params[1]#用于计算输出的切分次数
        
        if local_next == 1 and o_flag == 16: #输出复用为下一个算子的输入
            store_latency = 0
        else:#不复用
            store_latency = (o_params[0] * self.config["4X4 tile"]) / self.config["Dram_bandwidth"]
        return store_latency


if __name__ == "__main__":
    tx8_config = load_config("./hardware_parameter.json")
    arch = Tx8(tx8_config)

# # 创建Tx8类的实例
# arch = Tx8()

    # 使用Projection模块第一个RMSNorm算子验证其结果是否正确
    print("Projection的RMSNorm算子验证(后续不复用):")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([2,16], [2],[0.00048828,16], [[0.00390625,0]], 0, 0, 0, 1, 0)
    print("是否满足SRAM要求:", verification_result_0)
    print("总计算时间:", total_cp_latency_0)
    print("总通信时间:", total_cm_latency_0)
    print("总访存时间:", total_DRAM_0)
    print("总延迟:", latency_0)
    print("利用率:", Utilization_0)


    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2,16], [0.015625],[0.25,128], [[0.0625,1]], 0.25, 0, 1, 1, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
    
    # Projection模块的RMSNorm算子与一个GEMM算子融合并验证其结果是否正确
    print("Projection的RMSNorm与GEMM算子融合验证:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2.00048828,16], [0.015625],[0.25,128], [[0.00390625,0],[0.0625,1]], 0.25, 0, 1, 1, 5)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
