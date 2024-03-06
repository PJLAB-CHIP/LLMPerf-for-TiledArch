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

#算力10进制，存储二进制表示
#noc占比随着切分份数变多而变大
#10ns noc dram 100ns
def gemm_auto_opt_mapper(op,arch,Tm_Tn=None,fusion_op1=None,fusion_op2=None,details=False):
    '''
    gemm算子映射切分策略搜索,默认input_stationary
    '''
    if fusion_op1!=None and details:
        print('{} is fused with the last {}!'.format(op['name'],fusion_op1['name']))
    if fusion_op2!=None and details:
        print('{} is fused with the next {}!'.format(op['name'],fusion_op2['name']))

    max_utilization=0
    best_parall=[]
    best_latency=[]
    for stationary in ['input','weight']:
        if stationary=='input':
            dims=op['ishape']+[op['wshape'][-1]]#[b,m,k,n]输入维度为[b,m,k] 权重维度为[k,n] 输出维度为[b,m,n]
        else:
            dims=[1,op['wshape'][1],op['wshape'][0],op['ishape'][0]*op['ishape'][1]]#[1,n,k,b*m]输入维度为[1,n,k] 权重维度为[k,b*m] 输出维度为[1,n,b*m]
            print(dims)
        tile_num=arch.config['TILE_NUM']
        Nm=block_range(dims[1],min_block=tile_num)
        Nn=block_range(dims[3],min_block=tile_num)
        if Tm_Tn!=None:
            if stationary=='input':
                Nm,Nn=[math.ceil(dims[0]*dims[1]/Tm_Tn[0])],[math.ceil(dims[3]/Tm_Tn[1])]
            else:
                Nm,Nn=[math.ceil(dims[0]*dims[1]/Tm_Tn[1])],[math.ceil(dims[3]/Tm_Tn[0])]

        for nm in Nm:
            for nn in Nn:
                current_parall=[1,nm,1,nn]
                cp=[]
                newshape,ishape,oshape,wshape,reduce=dim_analysis('GEMM',dims,current_parall)
                i_size=MBytes(ishape)
                if fusion_op1!=None:
                    if fusion_op1['wshape']!=None:
                        i_size+=(MBytes(fusion_op1['wshape']))
                    cp.append([fusion_op1['compute']/nm,0])
                i_params=[i_size,nm]
                o_size=MBytes(oshape)
                cp.append([op['compute']/nm/nn,1])
                if fusion_op2!=None:
                    if fusion_op2['wshape']!=None:
                        o_size+=(MBytes(fusion_op2['wshape']))/nm/nn
                    cp.append([fusion_op2['compute']/nn/nm,0])
                o_params=[o_size,nm*nn]
                w_params=[MBytes(wshape),nn]
                cm_size,cm_type,cm_hops=w_params[0],0,5
                #print(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
                sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
                #print(arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops))
                if tot_utilization>max_utilization and sram_cap_req:
                    #print(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
                    max_utilization=tot_utilization
                    best_parall=current_parall
                    best_latency=tot_latency
    if  details:      
        print('{:<15}, dims={}, best={}, stationary={}'.format(op['name'],dims,best_parall,stationary))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result

def flashatten_mapper(model,arch,Tx_Ty=None,details=True,Head_fused=True):
    #将Q,KV分成特定的块数，同时将不同的块分配到tile上，每个tile上一块。其中Q视为input,K&V视为权重；
    #Q:[B,Tx,H/A,A] KV=[B,Ty,H/A,A] S=[B,Tx,H/A,A]
    #外层循环次数 S/Tx,内层循环次数 S/Ty，一轮内层循环结束才输出部分和的结果S=[B,Tx,H/A,A]，而不是内层循环次数*外层循环次数
    #Head_fused 表示是否多头输入Q预加载优化
    config=model.config
    dims=[config['B'],config['S'],int(config['H']/config['A']),config['A']]
    #print("config['A']",config['A'])
    Tx=block_range(dims[1],min_block=1,max_block=dims[1]//arch.config['TILE_NUM'])
    Ty=block_range(dims[1],min_block=1,max_block=dims[1]//arch.config['TILE_NUM'])
    if Tx_Ty!=None:
        assert Tx_Ty[0]<=dims[1]//arch.config['TILE_NUM']
        assert Tx_Ty[1]<=dims[1]//arch.config['TILE_NUM']
        Tx,Ty=[Tx_Ty[0]],[Tx_Ty[1]]
    max_utilization=0
    best_tx_ty=[]
    best_latency=0
    best_total_cp_latency=0
    for tx in Tx:#outer Q
        for ty in Ty:
            current_tx_ty=[tx,ty]
            Q_RoPE_wsize=model.config['Q']//8*tx*model.config['H']//model.config['A']/MB
            K_RoPE_wsize=model.config['Q']//8*ty*model.config['H']//model.config['A']/MB
            if Head_fused:
                head=dims[3]
            else:
                head=1
            i_params=[MBytes([dims[0],tx,dims[2]])+Q_RoPE_wsize,head*math.ceil(dims[1]//tx)]# 将多头也进行overlap，隐藏Q的输入时间
            o_params=[MBytes([dims[0],tx,dims[2]]),head*math.ceil(dims[1]//tx)]
            w_params=[2*MBytes([dims[0],ty,dims[2]])+K_RoPE_wsize,math.ceil(dims[1]//ty)]#K+V
            vector_cp_size=model.config['B']*tx*model.config['H']//model.config['A']+ model.config['B']*ty*model.config['H']//model.config['A']#RoPE
            flash_vector_cp_size=5*tx*ty#*dims[2]
            #cp=[[2*2*tx*ty*dims[2]/G,1]]
            #cp=[[0,0],[2*2*tx*ty*dims[2]/G,1],[0,0]]
            cp=[[vector_cp_size/G,0],[2*2*tx*ty*dims[2]/G,1],[flash_vector_cp_size/G,0]]
            cm_size,cm_type,cm_hops=w_params[0],0,1
            #print('test',i_params,o_params,w_params,cp,cm_size,cm_type,cm_hops)
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp,  cm_size, cm_type,cm_hops)
            #print('data',sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization)
            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_tx_ty=current_tx_ty
                best_latency=tot_latency
                best_total_cp_latency=total_cp_latency
                #print('test',i_params,o_params,w_params,cp,cm_size,cm_type,cm_hops)
                #print('data,current_tx_ty={},sram_cap_req={},total_cp_latency={},tot_latency={}, tot_utilization={}'.format(best_tx_ty,sram_cap_req,total_cp_latency,tot_latency, tot_utilization))
    if details:
        print('{:<15}, dims={}, best={}'.format('Flashatten',dims,best_tx_ty)) 
        if Head_fused:
            one_head_latency,one_head_cp_latency=best_latency/dims[3],best_total_cp_latency/dims[3]
        else:
            one_head_latency,one_head_cp_latency=best_latency,best_total_cp_latency

        print('latency={}, compute latency={}'.format(one_head_latency,one_head_cp_latency))
    #print(best_latency,best_total_cp_latency) 
    result={"latency":dims[3]//head*best_latency,'utilization':max_utilization,'cp_latency':dims[3]//head*best_total_cp_latency}
    return result

def vector_mapper(op,arch,splits=None,details=False):
    assert op['ishape']==op['oshape']
    io_shape,w_shape=op['ishape'],op['wshape']
    assert (op['name'] in ['RMSNorm','RMSNorm2','Hadamard','ResAdd','ResAdd2',]) and (op['type']=='Vector')
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
            o_params=[MBytes(io_shape)/split,split]
            w_params=[MBytes(w_shape)/split,split]#逐点运算输出切分数等于输入切分数，默认输出切分数=输入切分数*权重切分数
            cp=[[op['compute']/split,0]]
            #print(cp,split)
            cm_size,cm_type,cm_hops=0,0,0
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp,cm_size, cm_type,cm_hops)
            #print(sram_cap_req,total_cp_latency)
            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_split=split
                best_latency=tot_latency
    if details:
        print('{:<15}, best={}'.format(op['name'],best_split))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result

def manual_mapper(model,arch,QKV_fusion=True,preset=True,details=True):
    #指定映射
    Layers=model.config['L']
    ops=model.ops
    mapping_result={}
    if details:
        print('-'*40+'mapping_processing'+'-'*40)
    #1
    
    if QKV_fusion:
        ops["QKV_fusion"] =model.gen_gemm("QKV_fusion",[model.config["B"], model.config["S"], model.config["H"],3*model.config["H"]])
        Tm_Tn=[256,8] if preset else None
        #mapping_result['QKV_fusion']=gemm_auto_opt_mapper(ops['QKV_fusion'],arch,Tm_Tn=Tm_Tn,fusion_op1=None,details=details)
        mapping_result['QKV_fusion']=gemm_auto_opt_mapper(ops['QKV_fusion'],arch,Tm_Tn=Tm_Tn,fusion_op1=ops['RMSNorm'],details=details)
        del ops['Q_proj']
        del ops['K_proj']
        del ops['V_proj']
        del ops['RMSNorm']
    else:
        Tm_Tn=[256,32] if preset else None
        mapping_result['RMSNorm&Q_proj']=gemm_auto_opt_mapper(ops['Q_proj'],arch,Tm_Tn=Tm_Tn,fusion_op1=ops['RMSNorm'],details=details)
        mapping_result['K_proj']=gemm_auto_opt_mapper(ops['K_proj'],arch,Tm_Tn=Tm_Tn,details=details)
        mapping_result['V_proj']=gemm_auto_opt_mapper(ops['V_proj'],arch,Tm_Tn=Tm_Tn,details=details)
        del ops['RMSNorm']
        del ops['Q_proj']
    
    #2
    
    Tx_Ty=[256,256] if preset else None  #wanghuizheng
    mapping_result['Flashatten']=flashatten_mapper(model,arch,Tx_Ty=Tx_Ty,details=details,Head_fused=True)
    
    del ops['RoPE(Q)']
    del ops['RoPE(K)']
    del ops['QK^T']
    del ops['Softmax']
    del ops['AV']
    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,details=details)
    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],arch,splits=None,details=details)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],arch,splits=None,details=details)

    #3
    
    Tm_Tn=[32,256] if preset else None
    mapping_result['FFNup&SiLU']=gemm_auto_opt_mapper(ops['FFNup'],arch,Tm_Tn=Tm_Tn,fusion_op2=ops['SiLU'],details=details)
    del ops['SiLU']
    mapping_result['FFNgate']=gemm_auto_opt_mapper(ops['FFNgate'],arch,Tm_Tn=Tm_Tn,details=details)
    mapping_result['Hadamard']=vector_mapper(ops['Hadamard'],arch,splits=None)
    
    Tm_Tn=[4, 128] if preset else None
    mapping_result['FFNdown']=gemm_auto_opt_mapper(ops['FFNdown'],arch,Tm_Tn=Tm_Tn,details=details)
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
            print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(key,item['latency'],item['utilization']*100,item['cp_latency']))
        except:
            print('{:<15}, No suitable mapping result! '.format(key))
    mapping_result['Total']={"latency":tot_latency,'utilization':tot_cp_latency/tot_latency,'cp_latency':tot_cp_latency}
    print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format('Total Layers',tot_latency*Layers,tot_cp_latency/tot_latency*100,tot_cp_latency*Layers))
    
    return mapping_result 

if __name__ == "__main__":
    llm_config =load_config("./input/transformer/llama7b.json")
    llama7b = tbk.Llama_block(llm_config)
    print(llama7b.config)
    tx8_config=load_config('hardware_parameter.json')
    hardware=arch.Tx8(tx8_config)
    print(hardware.config)
    #preset 是否使用预设切分;details是否打印映射的详细信息
    mapping_result=manual_mapper(llama7b,hardware,preset=False,details=True)

