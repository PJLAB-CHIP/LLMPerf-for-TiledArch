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
    def execute(self, i_params, o_params, w_params, cp_size, cp_type, dram_size, dram_type, cm_size, cm_type, local_last, local_next, cm_hops, n):
        # i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
        # o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；o_flag为输出的总份数，例如16/32/64等
        # w_params = [w_size, w_flag, w_cm_flag] w_size为一份输出的大小，单位为MB；w_flag为输出的总份数；w_cm_flag为一轮内通信的次数，例如15次，则计算和存储是16次
        # cp_size为计算量，单位为GFLOPs
        # cp_type为计算类型, 这里认为0为Vector，1为Gemm
        # dram_size为DRAM读取数据大小，单位MB
        # dram_type
        # cm_size为通信量大小，单位MB
        # cm_type
        # local_last为上个算子的输出是否复用，0为不复用，1为复用
        # local_next为本算子的输出是否复用，0为不复用，1为复用
        # cm_hops为通信的最大跳数
        # n为流水的总次数，总延迟为开头+结尾+n次的最大延时（用于计算利用率），共n轮
        
        verification_result = self.verification(i_params, o_params, w_params)#要求(1)是否可以计算
        
        cp_latency = self.Computation_latency(cp_size, cp_type)#每一次的计算时间
        cp_latency_cycle = (w_params[2] + 1) * cp_latency#每一轮的计算时间
        total_cp_latency = n * cp_latency_cycle#总计算时间
        
        cm_latency_noc, overlap = self.Communication_latency(cm_size, cm_type, cm_hops)#每一次noc通信时间和overlap时间
        cm_latency = cm_latency_noc + 0.001 * overlap#每一次的通信时间，单位为ms
        cm_latency_cycle = w_params[2] * cm_latency#每一轮的通信时间
        total_cm_latency = n * cm_latency_cycle#总通信时间
        
        dram_read = self.DRAM_read(i_params, dram_size, dram_type, local_last)#每一轮DRAM读取数据的时间
        total_dram_read = n * dram_read#总DRAM读取时间
        
        dram_store = self.DRAM_store(o_params, dram_size, dram_type, local_next)#每一次DRAM存储数据的时间
        dram_store_cycle = (w_params[2] + 1) * dram_store#每一轮DRAM存储数据的时间
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
        
        return verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization
    
    def verification(self, i_params, o_params, w_params):
    # 该函数用于验证SRAM是否够用。如果切分数量小于等于16，则不需要遵循SRAM的2-3-2分配策略（2输入，3权重，2输出）
        if i_params[1] and o_params[1] <= 16 and w_params[1] <= 16 and i_params[1] == o_params[1]:
            if i_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
        
        elif i_params[1] and o_params[1] <= 16 and w_params[1] <= 16 and i_params[1] != o_params[1]:
            if i_params[0] + o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
                
        elif i_params[1] <= 16 and o_params[1] <= 16 and w_params[1] > 16:
            if i_params[0] + o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] <= 16 and o_params[1] > 16 and w_params[1] <= 16:
            if i_params[0] + 2 * o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] <= 16 and o_params[1] > 16 and w_params[1] > 16:
            if i_params[0] + 2 * o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] > 16 and o_params[1] <= 16 and w_params[1] <= 16:
            if 2 * i_params[0] + o_params[0] + w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False
 
        elif i_params[1] > 16 and o_params[1] <= 16 and w_params[1] > 16:
            if 2 * i_params[0] + o_params[0] + 3 * w_params[0] <= 3:
                verification_result = True
            else:
                verification_result = False

        elif i_params[1] > 16 and o_params[1] > 16 and w_params[1] <= 16:
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


    def Computation_latency(self, cp_size, cp_type):
        if cp_type == 0:
            cp_latency = cp_size / self.config["Vector"]#cp_type=0为Vector计算
        else:
            cp_latency = cp_size / self.config["Gemm"]#cp_type=1为GEMM计算
        return cp_latency


    def Communication_latency(self, cm_size, cm_type, cm_hops):
        cm_latency = cm_size / self.config["noc_bandwidth"]#通信延迟，单位为ms
        overlap = self.config["One_Hops"] * cm_hops#overlap用时，单位为us
        return cm_latency, overlap
    
    def DRAM_read(self, i_params, dram_size, dram_type, local_last):
        if local_last == 0:#不复用
            read_latency = (dram_size * self.config["4X4 tile"]) / self.config["Dram_bandwidth"]
        else:#复用
            read_latency = (dram_size - i_params[0]) * self.config["4X4 tile"] / self.config["Dram_bandwidth"]
        return read_latency
    
    def DRAM_store(self, o_params, dram_size, dram_type, local_next):
        if local_next == 1 and o_params[1] == 16: #输出复用为下一个算子的输入
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
    print("Projection的RMSNorm算子验证:")
    verification_result_0, total_cp_latency_0, total_cm_latency_0, total_DRAM_0, latency_0, Utilization_0 = arch.execute([2,16], [2,16],[0.0078125,1,0], 0.00390625, 0, 2.0078125, 0, 0, 0, 0, 1, 0, 1)
    print("是否满足SRAM要求:", verification_result_0)
    print("总计算时间:", total_cp_latency_0)
    print("总通信时间:", total_cm_latency_0)
    print("总访存时间:", total_DRAM_0)
    print("总延迟:", latency_0)
    print("利用率:", Utilization_0)


    # 使用Projection模块第一个GEMM算子验证其结果是否正确
    print("Projection的GEMM算子验证:")
    verification_result, total_cp_latency, total_cm_latency, total_DRAM, latency, Utilization = arch.execute([2,16], [0.015625,2048],[0.25,128,15], 0.0625, 1, 2.25, 0, 0.25, 0, 1, 1, 5, 8)
    print("是否满足SRAM要求:", verification_result)
    print("总计算时间:", total_cp_latency)
    print("总通信时间:", total_cm_latency)
    print("总访存时间:", total_DRAM)
    print("总延迟:", latency)
    print("利用率:", Utilization)
