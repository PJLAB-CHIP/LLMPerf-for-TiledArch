import transformer_block as tbk
import arch_execution as arch
from util import*
import math
from copy import deepcopy
  
def gemm_auto_opt_mapper(op,arch,local_last=True,local_next=True,input_stationary=True,elt_op=None):
    #elt_op 表示gemm算子后续紧跟着一个逐点非线性算子
    dims=op['ishape']+[op['wshape'][-1]]#[b,m,k,n]输入维度为[b,m,k] 权重维度为[k,n] 输出维度为[b,m,n]
    if input_stationary:
        N1=dim_split(dims[1],16)#时间维度展开
        N2=dim_split(dims[3])#空间维度展开
    else:
        N1=dim_split(dims[1])#时间维度展开
        N2=dim_split(dims[3],16)#空间维度展开
    # i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
    # o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；o_flag为输出的总份数，例如16/32/64等
    # w_params = [w_size, w_flag, w_cm_flag] w_size为一份输出的大小，单位为MB；w_flag为权重的总份数；w_cm_flag为一轮内通信的次数，例如15次，则计算和存储是16次
    # cp_size为计算量，单位为GFLOPs
    # cp_type为计算类型, 这里认为0为Vector，1为Gemm
    # dram_size为DRAM读取数据大小，单位MB
    # dram_type 0
    # cm_size为通信量大小，单位MB
    # cm_type 0
    # local_last为上个算子的输出是否复用，0为不复用，1为复用
    # local_next为本算子的输出是否复用，0为不复用，1为复用
    # cm_hops为通信的最大跳数
    # N为流水的总次数，总延迟为开头+结尾+N次的最大延时（用于计算利用率），共N轮,1轮代表片上权重数据遍历一次
    max_utilization=0
    best_parall=[]
    best_latency=[]
    for n1 in N1:
        for n2 in N2:
            #print(n1,n2)
            current_parall=[1,n1,1,n2]
            new_shape,i_shape,o_shape,w_shape,reduce=dim_analysis('GEMM',dims,current_parall)
            i_params=[MBytes(i_shape),n1]
            o_params=[MBytes(o_shape),n2]
            w_params=[MBytes(w_shape),n2*n1,n1-1]
            if elt_op!=None:
                elt_cp_size=elt_op['compute']*arch.vector/arch.gemm
            else:
                elt_cp_size=0
            cp_size,cp_type=(op['compute']+elt_cp_size)/n2/n1,1
            #TODO 删除dram_size
            dram_size,dram_type=MBytes(i_shape)+MBytes(w_shape)+MBytes(o_shape),0
            cm_size,cm_type,cm_hops,N=w_params[0],0,5,n1//n2
            
            _,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp_size, cp_type, dram_size, dram_type, cm_size, cm_type, local_last, local_next, cm_hops, N)
            if tot_utilization>max_utilization:
                max_utilization=tot_utilization
                best_parall=current_parall
                best_latency=tot_latency
            
    print('operator dims={},best_parallelism={},stationary={}'.format(dims,best_parall,'input' if input_stationary else 'weight'))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result


def vector_mapper(op,op_type,arch,split_num=1,local_last=True,local_next=True):
    #split_num 大小调整加载次数，使得与上游算子或者下游算子切分对齐，保证其数据局部性 
    assert op['ishape']==op['oshape']
    cp_type=1
    io_shape=op['ishape']
    w_shape=op['wshape']
    tile_num=16#arch.tile_num
    if op_type in ['RMSNorm','Hadamard','ResAdd'] : 
            #Resadd 输入和权重shape一样，因此可以不区分stationary
            i_params=[MBytes(io_shape)/tile_num/split_num,tile_num]
            o_params=[MBytes(io_shape)/tile_num/split_num,tile_num]
            w_params=[MBytes(w_shape),1,0]
            cp_size=op['compute']/tile_num/split_num
            #TODO 删除dram_size
            dram_size,dram_type=0,0
            if op_type=='ResAdd':
                hops=2
            else:
                hops=1
            cm_size,cm_type,cm_hops,N=0,0,hops,split_num

    _,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp_size, cp_type, dram_size, dram_type, cm_size, cm_type, local_last, local_next, cm_hops, N)
    result={"latency":tot_latency,'utilization':tot_utilization,'cp_latency':total_cp_latency}
    return result
  
def flashatten_mapper(model,arch,local_last=False,local_next=False):
    max_utilization=0
    best_tx_ty=[]
    best_latency=[]
    config=model.config
    dims=[config['B'],config['S'],int(config['H']/config['A']),config['A']]
    #将Q,KV分成特定的块数，同时将不同的块分配到tile上，每个tile上一块。其中Q视为input,K&V视为权重
    TX=dim_split(dims[1])
    TY=dim_split(dims[1])
    tile_num=arch.tile_num
    for tx in TX:
        for ty in TY:
            current_tx_ty=[tx,ty]
            i_params=[MBytes([dims[0],tx,dims[2]]),tile_num]
            o_params=[MBytes([dims[0],tx,ty]),math.ceil(dims[1]/ty)]
            w_params=[2*MBytes([dims[0],ty,dims[2]]),tile_num*math.ceil(dims[1]/ty),tile_num-1]#K+V
            elt_cp_size=0*arch.vector/arch.gemm#忽略增加的计算量
            cp_size,cp_type=2*tx*ty*dims[2]+elt_cp_size,1
            #TODO 删除dram_size
            dram_size,dram_type=0,0
            cm_size,cm_type,cm_hops=w_params[0],0,5
            N=math.ceil(MBytes(dims[0:-1])/tx/tile_num)
            _,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp_size, cp_type, dram_size, dram_type, cm_size, cm_type, local_last, local_next, cm_hops, N)
            if tot_utilization>max_utilization:
                max_utilization=tot_utilization
                best_tx_ty=current_tx_ty
                best_latency=tot_latency
    print('operator dims={},best_tx_ty={}'.format(dims,best_tx_ty))
    result={"latency":best_latency*dims[3],'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result

def manual_mapper(model,arch,qkv_fusion=False):
    #指定映射
    ops=model.ops
    mapping_result={}
    #1
    mapping_result['RMSNorm']=vector_mapper(ops['RMSNorm'],'RMSNorm',arch,split_num=1,local_last=False,local_next=True)
    
    #TODO qkv
    if qkv_fusion:
        ishape=[model.config["B"], model.config["S"], model.config["H"]]
        wshape = [model.config["H"], model.config["H"]]
        oshape = [ishape[0], ishape[1], wshape[1]]
        Proj_compute = 2*ishape[0]*ishape[1]*wshape[0]*wshape[1]/1024 * 1024 * 1024
        ops["QKV_proj"] = {"type": "GEMM", "ishape":ishape, "wshape": wshape, "oshape":oshape, "compute":Proj_compute}
        mapping_result['QKV_proj']=gemm_auto_opt_mapper(ops['QKV_proj'],arch,local_last=True,local_next=False)
        del ops['Q_proj']
        del ops['K_proj']
        del ops['V_proj']
    else:
        mapping_result['Q_proj']=gemm_auto_opt_mapper(ops['Q_proj'],arch,local_last=True,local_next=False)
        mapping_result['K_proj']=gemm_auto_opt_mapper(ops['K_proj'],arch,local_last=False,local_next=False)
        mapping_result['V_proj']=gemm_auto_opt_mapper(ops['V_proj'],arch,local_last=False,local_next=False)
    #2
    mapping_result['Flashatten']=flashatten_mapper(model,arch,local_last=False,local_next=False)
    del ops['RoPE(Q)']
    del ops['RoPE(K)']
    del ops['QK^T']
    del ops['Softmax']
    del ops['AV']


    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,local_last=True,local_next=False)
    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],'RMSNorm',arch,split_num=1,local_last=False,local_next=True)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],'ResAdd',arch,local_last=True,local_next=False)
    #3
    mapping_result['FFN_up&SiLU']=gemm_auto_opt_mapper(ops['FFN_up'],arch,local_last=True,local_next=False,elt_op=ops['SiLU'])
    del ops['SiLU']

    mapping_result['FFN_gate']=gemm_auto_opt_mapper(ops['Q_proj'],arch,local_last=True,local_next=False)
    mapping_result['hadamard']=vector_mapper(ops['hadamard'],'hadamard',arch,split_num=1,local_last=False,local_next=True)
    mapping_result['FFN2']=gemm_auto_opt_mapper(ops['FFN2'],arch,local_last=True,local_next=False)
    mapping_result['ResAdd2']=vector_mapper(ops['ResAdd2'],'ResAdd',arch,split_num=1,local_last=False,local_next=True)
    
    tot_latency=0
    tot_cp_latency=0
    tot_utilization=0
    for item in mapping_result.items():
        tot_latency+=item['latency']
        tot_cp_latency+=item['cp_latency']
        tot_utilization+=item['utilization']
    mapping_result['Total']={"latency":tot_latency,'utilization':tot_cp_latency/tot_latency,'cp_latency':tot_cp_latency}
    
    return mapping_result

if __name__ == "__main__":
    llm_config =load_config("./input/transformer/input0.json")
    llama7b = tbk.Llama_block(llm_config)
    print(llama7b.config)
    tx8_config=load_config('hardware_parameter.json')
    hardware==arch.Tx8(tx8_config)
    print(hardware.config)
    mapping_result=manual_mapper(llama7b,hardware)
    print(mapping_result)
