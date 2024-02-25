import transformer_block as tbk
import arch_execution as arch
from util import*
import math
# fusion_op 表示gemm算子与vector算子融合
# i_params = [i_size, i_flag] i_size为一份输入的大小，单位为MB；i_flag为输入的总份数，例如16/32/64等
# o_params = [o_size, o_flag] o_size为一份输出的大小，单位为MB；o_flag为输出的总份数，例如16/32/64等
# w_params = [w_size, w_flag] w_size为一份输出的大小，单位为MB；w_flag为权重的总份数；w_cm_flag为一轮内通信的次数，例如15次，则计算和存储是16次
# cp=[[cp_size,cp_type],...]为计算量，单位为GFLOPs, cp_type为计算类型, 这里认为0为Vector，1为Gemm
# cm_size为通信量大小，单位MB,cm_type 0,cm_hops为通信的最大跳数 
def gemm_auto_opt_mapper(op,arch,input_stationary=True,Nm_Nn=None,fusion_op1=None,fusion_op2=None,details=False):
    '''
    gemm算子映射切分策略搜索,默认input_stationary
    '''
    if fusion_op1!=None and details:
        print('{} is fused with the last {}!'.format(op['name'],fusion_op1['name']))
    if fusion_op2!=None and details:
        print('{} is fused with the next {}!'.format(op['name'],fusion_op2['name']))
    if input_stationary:
        dims=op['ishape']+[op['wshape'][-1]]#[b,m,k,n]输入维度为[b,m,k] 权重维度为[k,n] 输出维度为[b,m,n]
    else:
        dims=[1]+op['wshape']+[op['ishape'][0]*op['ishape'][1]]#[1,n,k,b*m]输入维度为[1,n,k] 权重维度为[k,b*m] 输出维度为[1,n,b*m]
        #print(dims)
    tile_num=arch.config['4X4 tile']
    Nm=block_range(dims[1],min_block=tile_num)
    Nn=block_range(dims[3],min_block=tile_num)
    if Nm_Nn!=None:
        Nm,Nn=[Nm_Nn[0]],[Nm_Nn[1]]
    max_utilization=0
    best_parall=[]
    best_latency=[]
    for nm in Nm:
        for nn in Nn:
            current_parall=[1,nm,1,nn]
            cp=[]
            newshape,ishape,oshape,wshape,reduce=dim_analysis('GEMM',dims,current_parall)
            i_size=MBytes(ishape)
            if fusion_op1!=None:
                if fusion_op1['wshape']!=None:
                    i_size+=(MBytes(fusion_op1['wshape']))/nm
                cp.append([fusion_op1['compute']/nm,0])
            i_params=[i_size,nm]
            o_params=[MBytes(oshape)]
            w_params=[MBytes(wshape),nn]
            cp.append([op['compute']/nm/nn,1])
            if fusion_op2!=None:
                if fusion_op2['wshape']!=None:
                    o_size+=(MBytes(fusion_op2['wshape']))/nm/nn
                cp.append([fusion_op2['compute']/nn/nm,0])
            cm_size,cm_type,cm_hops=w_params[0],0,5
            #print(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
            #print(arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops))
            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_parall=current_parall
                best_latency=tot_latency
    if  details:      
        print('{:<15}, dims={}, best={}, stationary={}'.format(op['name'],dims,best_parall,'input' if input_stationary else 'weight'))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result

def flashatten_mapper(model,arch,Tx_Ty=None,details=False):
    #将Q,KV分成特定的块数，同时将不同的块分配到tile上，每个tile上一块。其中Q视为input,K&V视为权重
    GB=1024*1024*1024
    config=model.config
    dims=[config['B'],config['S'],int(config['H']/config['A']),config['A']]
    Tx=block_range(dims[1],min_block=1)
    Ty=block_range(dims[1],min_block=1)
    if Tx_Ty!=None:
        Tx,Ty=[Tx_Ty[0]],[Tx_Ty[1]]

    max_utilization=0
    best_tx_ty=[]
    best_latency=[]
    for tx in Tx:#outer Q
        for ty in Ty:
            current_tx_ty=[tx,ty]
            i_params=[MBytes([dims[0],tx,dims[2]]),math.ceil(dims[1]//tx)]
            o_params=[MBytes([dims[0],tx,ty])]
            w_params=[2*MBytes([dims[0],ty,dims[2]]),math.ceil(dims[1]//ty)]#K+V
            elt_cp_size=0*arch.config['Vector']/arch.config['Gemm']#增加的计算量(忽略)
            cp=[[(2*tx*ty*dims[2]+elt_cp_size)/GB,1]]
            cm_size,cm_type,cm_hops=w_params[0],0,1
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp,  cm_size, cm_type,cm_hops)

            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_tx_ty=current_tx_ty
                best_latency=tot_latency
    if details:
        print('{:<15}, dims={}, best={}'.format('Flashatten',dims,best_tx_ty)) 
    result={"latency":best_latency*dims[3],'utilization':max_utilization,'cp_latency':total_cp_latency*dims[3]}
    return result

def vector_mapper(op,arch,splits=None,details=False):
    assert op['ishape']==op['oshape']
    io_shape,w_shape=op['ishape'],op['wshape']
    assert (op['name'] in ['RMSNorm','RMSNorm2','Hadamard','ResAdd','ResAdd2']) and (op['type']=='Vector')
    i_split=op['ishape'][1]#RMS只能切一个维度
    if splits==None:
        if op['name'] in ['Hadamard','ResAdd','ResAdd2']:
            i_split=i_split*op['ishape'][2]
        splits=block_range(i_split,min_block=1)
    else:
        splits=[splits]
    max_utilization=0
    best_split=[]
    best_latency=[]
    for split in splits:
            i_params=[MBytes(io_shape)/split,split]
            o_params=[MBytes(io_shape)/split]
            w_params=[MBytes(w_shape)/split,1]#逐点运算输出切分数等于输入切分数，默认输出切分数=输入切分数*权重切分数
            cp=[[op['compute']/split,0]]
            cm_size,cm_type,cm_hops=0,0,0
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp,cm_size, cm_type,cm_hops)
            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_split=split
                best_latency=tot_latency
    if details:
        print('{:<15}, best={}'.format(op['name'],best_split))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result

def manual_mapper(model,arch,QKV_fusion=True,Preset=True,details=True):
    #指定映射
    Layers=model.config['L']
    ops=model.ops
    mapping_result={}
    if details:
        print('-'*40+'mapping_processing'+'-'*40)
    #1
    if QKV_fusion:
        ops["QKV_fusion"] =model.gen_gemm("QKV_fusion",[model.config["B"], model.config["S"], model.config["H"],3*model.config["H"]])
        Nm_Nn=[32,512] if Preset else None
        mapping_result['QKV_fusion']=gemm_auto_opt_mapper(ops['QKV_fusion'],arch,Nm_Nn=[32,512],fusion_op1=ops['RMSNorm'],details=details)
        del ops['Q_proj']
        del ops['K_proj']
        del ops['V_proj']
        del ops['RMSNorm']
    else:
        Nm_Nn=[16,128] if Preset else None
        mapping_result['RMSNorm&Q_proj']=gemm_auto_opt_mapper(ops['Q_proj'],arch,Nm_Nn=Nm_Nn,fusion_op1=ops['RMSNorm'],details=details)
        mapping_result['K_proj']=gemm_auto_opt_mapper(ops['K_proj'],arch,Nm_Nn=Nm_Nn,details=details)
        mapping_result['V_proj']=gemm_auto_opt_mapper(ops['V_proj'],arch,Nm_Nn=Nm_Nn,details=details)
        del ops['RMSNorm']
        del ops['Q_proj']
    
    #2
    Tx_Ty=[256,256] if Preset else None  #wanghuizheng
    mapping_result['Flashatten']=flashatten_mapper(model,arch,Tx_Ty=Tx_Ty,details=details)
    del ops['RoPE(Q)']
    del ops['RoPE(K)']
    del ops['QK^T']
    del ops['Softmax']
    del ops['AV']
    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,details=details)
    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],arch,splits=None,details=details)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],arch,splits=None,details=details)

    #3
    Nm_Nn=[128,128] if Preset else None
    mapping_result['FFNup&SiLU']=gemm_auto_opt_mapper(ops['FFNup'],arch,Nm_Nn=Nm_Nn,fusion_op2=ops['SiLU'],details=details)
    del ops['SiLU']
    mapping_result['FFNgate']=gemm_auto_opt_mapper(ops['FFNgate'],arch,Nm_Nn=Nm_Nn,details=details)
    mapping_result['Hadamard']=vector_mapper(ops['Hadamard'],arch,splits=None)

    Nm_Nn=[256,128] if Preset else None
    mapping_result['FFN2']=gemm_auto_opt_mapper(ops['FFN2'],arch,Nm_Nn=Nm_Nn,details=details)
    mapping_result['ResAdd2']=vector_mapper(ops['ResAdd2'],arch,splits=None,details=details)

    print('-'*40+'mapping_result'+'-'*40)
    tot_latency=0
    tot_cp_latency=0
    tot_utilization=0
    for key,item in mapping_result.items():
        try:
            tot_latency+=item['latency']
            tot_cp_latency+=item['cp_latency']
            tot_utilization+=item['utilization']
            print('{:<15}, latency(ms)={:>10.4f}, utilization(%)={:>10.4f}, compute latency(ms)={:>10.4f}'.format(key,item['latency'],item['utilization']*100,item['cp_latency']))
        except:
            print('{:<15}, No suitable mapping result! '.format(key))
    mapping_result['Total']={"latency":tot_latency,'utilization':tot_cp_latency/tot_latency,'cp_latency':tot_cp_latency}
    print('{:<15}, latency(ms)={:>10.4f}, utilization(%)={:>10.4f}, compute latency(ms)={:>10.4f}'.format('Total Layers',tot_latency*Layers,tot_cp_latency/tot_latency*100,tot_cp_latency*Layers))
    return mapping_result

if __name__ == "__main__":
    llm_config =load_config("./input/transformer/input0.json")
    llama7b = tbk.Llama_block(llm_config)
    print(llama7b.config)
    tx8_config=load_config('hardware_parameter.json')
    hardware=arch.Tx8(tx8_config)
    print(hardware.config)
    mapping_result=manual_mapper(llama7b,hardware,Preset=True,details=True)
    #print(mapping_result)
