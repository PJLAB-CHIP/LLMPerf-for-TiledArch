import transformer_block as tbk
import arch_execution as arch
from util import*
import math
from mapper import *
from gemm_tiling import *
# fusion_op 表示gemm算子与vector算子融合
# i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
# o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；o_flag为输出的总份数，例如16/32/64等
# w_params = [w_size, w_flag] w_size为一份输出的大小，单位为MB；w_flag为权重的总份数；w_cm_flag为一轮内通信的次数，例如15次，则计算和存储是16次
# cp=[[cp_size,cp_type],...]为计算量，单位为GFLOPs, cp_type为计算类型, 这里认为0为Vector，1为Gemm
# cm_size为通信量大小，单位MB,cm_type 0,cm_hops为通信的最大跳数 
def test_mapper(model,arch,details=True):
    #ops=model.ops
    ops={}
    mapping_result={}
    M, K , N = 4096, 4096, 4096
    QKV = 1   # QKV = 3, if fuse q k v into one matrix
    N = N * QKV
    B = 1
    # B = 1
    ops["test_gemm"] =model.gen_gemm("test_gemm",[B, M, K,N])
    print(ops)
    tile_m_tile_n_g=[[128,64],[256,32],[128,32],[32,128],[64,64],[256,32],[64,128]]
    for tile_m_tile_n in tile_m_tile_n_g:
        print('1'*80)
        tile_m,tile_n=tile_m_tile_n
        Tm_Tn=tile_m_tile_n
        
        utilization = gemm_tiling_input_stationary(B, M, K, N, tile_m, tile_n, print_details=details)
        print(f"test_gemm_daixu, M={M}, K={K}, N={N}, B={B}, tile_m={tile_m}, tile_n={tile_n}, stationary: input, utilization={utilization:.2f}%")
        mapping_result['test_gemm']=gemm_auto_opt_mapper(ops['test_gemm'],arch,input_stationary = True, Tm_Tn=Tm_Tn,details=details)
        print('test_gemm,latency={},cp_latency={},utilization={:.2f}%'.format(mapping_result['test_gemm']['latency'],mapping_result['test_gemm']['cp_latency'],mapping_result['test_gemm']['utilization']*100))
        print('-'*80)
        
        utilization = gemm_tiling_weight_stationary(B, M, K, N, tile_m, tile_n, print_details=details)
        print(f"test_gemm_daixu, M={M}, K={K}, N={N}, B={B}, tile_m={tile_m}, tile_n={tile_n}, stationary: weight, utilization={utilization:.2f}%")
        mapping_result['test_gemm']=gemm_auto_opt_mapper(ops['test_gemm'],arch,input_stationary = False, Tm_Tn=Tm_Tn,details=details)
        print('test_gemm,latency={},cp_latency={},utilization={:.2f}%'.format(mapping_result['test_gemm']['latency'],mapping_result['test_gemm']['cp_latency'],mapping_result['test_gemm']['utilization']*100))
       
    #2
    '''
    Tx_Ty=[256,256] if preset else None  #wanghuizheng
    mapping_result['Flashatten']=flashatten_mapper(model,arch,Tx_Ty=Tx_Ty,details=details)
    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,details=details)
    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],arch,splits=None,details=details)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],arch,splits=None,details=details)
    '''


if __name__ == "__main__":
    llm_config =load_config("./input/transformer/input0.json")
    llama7b = tbk.Llama_block(llm_config)
    tx8_config=load_config('hardware_parameter.json')
    hardware=arch.Tx8(tx8_config)
    print(hardware.config)
    #preset 是否使用预设切分;details是否打印映射的详细信息
    mapping_result=test_mapper(llama7b,hardware, details=False)

