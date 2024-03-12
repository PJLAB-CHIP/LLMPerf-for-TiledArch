import transformer_block as tbk
from arch_execution import Tx8
from util import *
import math
# fusion_op 表示gemm算子与vector算子融合
# 算力10进制，存储二进制表示
def gemm_auto_opt_mapper(op,arch,TmTnTk=None,fusion_op1=None,fusion_op2=None,details=False):
    '''gemm算子映射切分策略搜索,默认input_stationary'''
    # i_params = [i_size, nm,nk] i_size为一份输入的大小，单位为MB；nm*nk为输入的总份数
    # o_params = [o_size, nn*nm] o_size为一份输出的大小，单位为MB；nn*nm为输出的总份数
    # w_params = [w_size, nn,nk] w_size为一份输出的大小，单位为MB；nn*nk为权重的总份数；
    # cp=[[cp_size,cp_type],...]为计算量，单位为GFLOPs, cp_type为计算类型, 这里认为0为Vector，1为Gemm
    # cm_size为通信量大小，单位MB,cm_type 0,cm_hops为通信的最大跳数 
    if fusion_op1!=None and details:
        print('{} is fused with the last {}!'.format(op['name'],fusion_op1['name']))
    if fusion_op2!=None and details:
        print('{} is fused with the next {}!'.format(op['name'],fusion_op2['name']))
    max_utilization=0
    best_parall=[]
    best_latency=[]
    best_stationary=None
    total_cp_latency = 0
    for stationary in ['input','weight']:
        if stationary=='input':
            dims=op['ishape']+[op['wshape'][-1]]#[b,m,k,n]输入维度为[b,m,k] 权重维度为[k,n] 输出维度为[b,m,n]
        else:
            dims=[1,op['wshape'][1],op['wshape'][0],op['ishape'][0]*op['ishape'][1]]#[1,n,k,b*m]输入维度为[1,n,k] 权重维度为[k,b*m] 输出维度为[1,n,b*m]
            #print(dims)
        tile_num=arch.config['TILE_NUM']
        dims=[dims[0]]+dim_norm(dims[1:],tile_num=tile_num)
        #print(dims)
        if TmTnTk!=None:
            if stationary=='input':
                Nm,Nn,Nk=[math.ceil(dims[0]*dims[1]/TmTnTk[0])],[math.ceil(dims[3]/TmTnTk[1])],[math.ceil(dims[2]/TmTnTk[2])] if len(TmTnTk)==3 else [1]
                #print(TmTnTk,dims,Nm,Nn,Nk)
            else:
                Nm,Nn,Nk=[math.ceil(dims[0]*dims[1]/TmTnTk[1])],[math.ceil(dims[3]/TmTnTk[0])],[math.ceil(dims[2]/TmTnTk[2])] if len(TmTnTk)==3 else [1]
                #print(TmTnTk,dims,Nm,Nn,Nk)
        else:
            Nm=block_range(dims[1],min_block=tile_num)
            Nk=block_range(dims[2])
            Nn=block_range(dims[3],min_block=tile_num)
        for nk in Nk:
            for nm in Nm:
                for nn in Nn:
                    cur_gemm_parall=[1,nm,nk,nn]
                    cp=[]
                    newdims,ishape,oshape,wshape,reduce=dim_analysis('GEMM',dims,cur_gemm_parall)
                    #print(newshape)
                    i_size,w_size,o_size=MBytes(ishape),MBytes(wshape),MBytes(oshape)
                    if fusion_op1!=None:
                        i_size+=MBytes(fusion_op1['wshape'])/nm/nk
                        cp.append([fusion_op1['compute']/nm/nk,0])
                    i_params=[i_size,nm,nk]
                    w_params=[w_size,nn,nk]
                    cp.append([op['compute']/nm/nn/nk,1])
                    #t=op['compute']/nm/nn
                    #print(nm,nn)
                    if fusion_op2!=None:
                        o_size+=(MBytes(fusion_op2['wshape'])/nm/nn)
                        cp.append([fusion_op2['compute']/nn/nm,0])
                    o_params=[o_size,nm*nn]
                    cm_size,cm_type,cm_hops=w_params[0],0,5
                    #print(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops)
                    sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops,details)
                    #print(arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops))
                    #print("total_cp_latency",total_cp_latency)
                    if tot_utilization>max_utilization and sram_cap_req:
                        max_utilization=tot_utilization
                        best_parall=cur_gemm_parall
                        best_latency=tot_latency
                        best_stationary=stationary
    if  details:      
        print('{:<15}, dims={}, best={}, stationary={}'.format(op['name'],dims,best_parall,best_stationary))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':total_cp_latency}
    return result


def flashatten_mapper(model, arch, Tx_Ty=None, details=True, Head_fused=True):
    # 将Q,KV分成特定的块数，同时将不同的块分配到tile上，每个tile上一块。其中Q视为input,K&V视为权重；
    # Q:[B,Tx,H/A,A] KV=[B,Ty,H/A,A] S=[B,Tx,H/A,A]
    # 外层循环次数 S/Tx,内层循环次数 S/Ty，一轮内层循环结束才输出部分和的结果S=[B,Tx,H/A,A]，而不是内层循环次数*外层循环次数
    # Head_fused 表示是否多头输入Q预加载优化
    config = model.config
    dims = [config['B'], config['S'], int(
        config['H']/config['A']), config['A']]
    # print("config['A']",config['A'])
    Tx = block_range(dims[1], min_block=1,
                     max_block=dims[1]//arch.config['TILE_NUM'])
    Ty = block_range(dims[1], min_block=1,
                     max_block=dims[1]//arch.config['TILE_NUM'])
    if Tx_Ty != None:
        assert Tx_Ty[0] <= dims[1]//arch.config['TILE_NUM']
        assert Tx_Ty[1] <= dims[1]//arch.config['TILE_NUM']
        Tx, Ty = [Tx_Ty[0]], [Tx_Ty[1]]
    max_utilization = 0
    best_tx_ty = []
    best_latency = 0
    best_total_cp_latency = 0
    for tx in Tx:  # outer Q
        for ty in Ty:
            current_tx_ty = [tx, ty]
            Q_RoPE_wsize = model.config['Q']//8*tx * \
                model.config['H']//model.config['A']/MB
            K_RoPE_wsize = model.config['Q']//8*ty * \
                model.config['H']//model.config['A']/MB
            if Head_fused:
                head = dims[3]
            else:
                head = 1
            i_params = [MBytes([dims[0], tx, dims[2]])+Q_RoPE_wsize,
                        head*math.ceil(dims[1]//tx)]  # 将多头也进行overlap，隐藏Q的输入时间
            o_params = [MBytes([dims[0], tx, dims[2]]),
                        head*math.ceil(dims[1]//tx)]
            w_params = [2*MBytes([dims[0], ty, dims[2]]) +
                        K_RoPE_wsize, math.ceil(dims[1]//ty)]  # K+V
            vector_cp_size = model.config['B']*tx*model.config['H']//model.config['A'] + \
                model.config['B']*ty * \
                model.config['H']//model.config['A']  # RoPE
            flash_vector_cp_size = 5*tx*ty  # *dims[2]
            # cp=[[2*2*tx*ty*dims[2]/G,1]]
            # cp=[[0,0],[2*2*tx*ty*dims[2]/G,1],[0,0]]
            cp = [[vector_cp_size/G, 0], [2*2*tx*ty*dims[2]/G, 1],
                  [flash_vector_cp_size/G, 0]]
            cm_size, cm_type, cm_hops = w_params[0], 0, 1
            # print('test',i_params,o_params,w_params,cp,cm_size,cm_type,cm_hops)
            sram_cap_req, total_cp_latency, _, _, tot_latency, tot_utilization = arch.execute(
                i_params, o_params, w_params, cp,  cm_size, cm_type, cm_hops,details)
            # print('data',sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization)
            if tot_utilization > max_utilization and sram_cap_req:
                max_utilization = tot_utilization
                best_tx_ty = current_tx_ty
                best_latency = tot_latency
                best_total_cp_latency = total_cp_latency
                # print('test',i_params,o_params,w_params,cp,cm_size,cm_type,cm_hops)
                # print('data,current_tx_ty={},sram_cap_req={},total_cp_latency={},tot_latency={}, tot_utilization={}'.format(best_tx_ty,sram_cap_req,total_cp_latency,tot_latency, tot_utilization))
    if details:
        print('{:<15}, dims={}, best={}'.format(
            'Flashatten', dims, best_tx_ty))
        if Head_fused:
            one_head_latency, one_head_cp_latency = best_latency / \
                dims[3], best_total_cp_latency/dims[3]
        else:
            one_head_latency, one_head_cp_latency = best_latency, best_total_cp_latency

        print('latency={}, compute latency={}'.format(
            one_head_latency, one_head_cp_latency))
    # print(best_latency,best_total_cp_latency)
    result = {"latency": dims[3]//head*best_latency, 'utilization': max_utilization,
              'cp_latency': dims[3]//head*best_total_cp_latency}
    return result

def vector_mapper(op,arch,splits=None,details=False):
    assert op['ishape']==op['oshape']
    io_shape,w_shape=op['ishape'],op['wshape']
    assert (op['name'] in ['RMSNorm','RMSNorm2','Hadamard','ResAdd','ResAdd2','SiLU',]) and (op['type']=='Vector')
    i_split=op['ishape'][1]#RMS只能切一个维度
    if splits==None:
        if op['name'] in ['Hadamard','ResAdd','ResAdd2','SiLU']:
            i_split=i_split*op['ishape'][2]
        splits=block_range(i_split,min_block=1)
    else:
        splits = [splits]
    max_utilization = 0
    best_split = []
    best_latency = []
    for split in splits:
            i_params=[MBytes(io_shape)/split,split]
            o_params=[MBytes(io_shape)/split,split]
            w_params=[MBytes(w_shape)/split,split]#逐点运算输出切分数等于输入切分数，默认输出切分数=输入切分数*权重切分数
            cp=[[op['compute']/split,0]]
            #print(op['compute'],op['compute']/split)
            cm_size,cm_type,cm_hops=0,0,0
            sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute(i_params, o_params, w_params, cp,cm_size, cm_type,cm_hops,details)
            #print(sram_cap_req,total_cp_latency)
            if tot_utilization>max_utilization and sram_cap_req:
                max_utilization=tot_utilization
                best_split=split
                best_latency=tot_latency
    if details:
        print('{:<15}, best={}'.format(op['name'], best_split))
    result = {"latency": best_latency,
              'utilization': max_utilization, 'cp_latency': total_cp_latency} 
    return result


def manual_mapper(model, arch, QKV_fusion=True, preset=True, details=True):
    # 指定映射
    Layers = model.config['L']
    ops = model.ops
    mapping_result = {}
    if details:
        print('-'*40+'mapping_processing'+'-'*40)
    #1
    if QKV_fusion:
        ops["QKV_fusion"] = model.gen_gemm("QKV_fusion", [
                                           model.config["B"], model.config["S"], model.config["H"], 3*model.config["H"]])
        TmTnTk = [256, 8] if preset else None
        # mapping_result['QKV_fusion']=gemm_auto_opt_mapper(ops['QKV_fusion'],arch,TmTnTk=TmTnTk,fusion_op1=None,details=details)
        mapping_result['QKV_fusion'] = gemm_auto_opt_mapper(
            ops['QKV_fusion'], arch, TmTnTk=TmTnTk, fusion_op1=ops['RMSNorm'], details=details)
        del ops['Q_proj']
        del ops['K_proj']
        del ops['V_proj']
        del ops['RMSNorm']
    else:
        TmTnTk = [256, 32] if preset else None
        mapping_result['RMSNorm&Q_proj'] = gemm_auto_opt_mapper(
            ops['Q_proj'], arch, TmTnTk=TmTnTk, fusion_op1=ops['RMSNorm'], details=details)
        mapping_result['K_proj'] = gemm_auto_opt_mapper(
            ops['K_proj'], arch, TmTnTk=TmTnTk, details=details)
        mapping_result['V_proj'] = gemm_auto_opt_mapper(
            ops['V_proj'], arch, TmTnTk=TmTnTk, details=details)
        del ops['RMSNorm']
        del ops['Q_proj']
    
    # 2

    Tx_Ty = [256, 256] if preset else None  # wanghuizheng
    mapping_result['Flashatten'] = flashatten_mapper(
        model, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)
    
    del ops['RoPE(Q)']
    del ops['RoPE(K)']
    del ops['QK^T']
    del ops['Softmax']
    del ops['AV']
    
    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,details=details)
    
    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],arch,splits=None,details=details)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],arch,splits=None,details=details)
    #3
    TmTnTk=None# [16,256] #if preset else None
    mapping_result['FFNup&SiLU']=gemm_auto_opt_mapper(ops['FFNup'],arch,TmTnTk=TmTnTk,fusion_op2=ops['SiLU'],details=details)
    del ops['SiLU']
    mapping_result['FFNgate'] = gemm_auto_opt_mapper(
        ops['FFNgate'], arch, TmTnTk=TmTnTk, details=details)
    mapping_result['Hadamard'] = vector_mapper(
        ops['Hadamard'], arch, splits=None)

    TmTnTk = [4, 128] if preset else None
    mapping_result['FFNdown'] = gemm_auto_opt_mapper(
        ops['FFNdown'], arch, TmTnTk=TmTnTk, details=details)
    mapping_result['ResAdd2'] = vector_mapper(
        ops['ResAdd2'], arch, splits=None, details=details)
   
    print('-'*40+'mapping_result'+'-'*40)
    tot_latency = 0
    tot_cp_latency = 0
    tot_utilization = 0
    utilization=0
    for key, item in mapping_result.items():
        try:
            tot_latency += item['latency']
            tot_cp_latency += item['cp_latency']
            tot_utilization += item['utilization']
            print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(
                key, item['latency'], item['utilization']*100, item['cp_latency']))
        except:
            print('{:<15}, No suitable mapping result! '.format(key))
    utilization=tot_cp_latency/(tot_latency+1e-35)
    mapping_result['Total'] = {
        "latency": tot_latency, 'utilization':utilization , 'cp_latency': tot_cp_latency}
    print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(
        'Total Layers', tot_latency*Layers, tot_cp_latency/tot_latency*100, tot_cp_latency*Layers))

    return mapping_result


if __name__ == "__main__":
    llm_config =load_config("./input/transformer/input0.json")
    #llm_config =load_config("./input/transformer/llama7b.json")
    llama7b = tbk.Llama_block(llm_config)
    print(llama7b.config)
    tx8_config = load_config('hardware_parameter.json')
    hardware = Tx8(tx8_config)
    print(hardware.config)
    # preset 是否使用预设切分;details是否打印映射的详细信息
    mapping_result = manual_mapper(
        llama7b, hardware, preset=False, details=False)
