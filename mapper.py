import tiled_arch_perf.transformer_block as tbk
from tiled_arch_perf.arch_execution import Tx8
from tiled_arch_perf.util import *
import math
import ipdb
# fusion_op 表示gemm算子与vector算子融合
# 算力10进制，存储二进制表示
def gemm_auto_opt_mapper(op,arch,TmTn=None,Tk=-1,fusion_op1=None,fusion_op2=None,details=False):
    '''gemm算子映射切分策略搜索,默认input_stationary'''
    #TmTn 代表M,N维度的size
    #Tk=-1 代表Reduce维度的size，None 代表不切该维度，-1代表自动搜索，other代表具体K维度size
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
    best_latency=0
    best_cp_latency=0
    best_stationary=None
    total_cp_latency = 0
    gemm_size = 64 # hardware support gemm of size 64
    for stationary in ['input','weight']:
        if stationary=='input':
            dims=op['ishape']+[op['wshape'][-1]]#[b,m,k,n]输入维度为[b,m,k] 权重维度为[k,n] 输出维度为[b,m,n]
        else:
            dims=[1,op['wshape'][1],op['wshape'][0],op['ishape'][0]*op['ishape'][1]]#[1,n,k,b*m]输入维度为[1,n,k] 权重维度为[k,b*m] 输出维度为[1,n,b*m]
            #print(dims)
        tile_num=arch.config['TILE_NUM']
        dims=[dims[0]]+dim_norm(dims[1:],tile_num=tile_num)  # 输出矩阵的m, n都padding到tile_num的倍数
        #print(dims)
        if TmTn!=None:
            if stationary=='input':
                Nm,Nn=[math.ceil(dims[0]*dims[1]/TmTn[0])],[math.ceil(dims[3]/TmTn[1])]
            else:
                Nm,Nn=[math.ceil(dims[0]*dims[1]/TmTn[1])],[math.ceil(dims[3]/TmTn[0])]
        else:
            Nm=split_range(dims[1],gemm_size=64*tile_num)
            Nn=split_range(dims[3],gemm_size=64*tile_num)
        if Tk==None:
            Nk=[1]
        elif Tk>0:
            Nk=[math.ceil(dims[2]/Tk)]
        else:
            Nk=block_range(dims[2])
        #ipdb.set_trace()
        for nk in Nk:
            for _nm in Nm:
                for _nn in Nn:
                    nm=_nm*tile_num
                    nn=_nn*tile_num
                    cur_gemm_parall=[1,nm,nk,nn]
                    cp=[]
                    #print(dims,cur_gemm_parall)
                    newdims,ishape,oshape,wshape,reduce=dim_analysis('GEMM',dims,cur_gemm_parall)
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
                    #print(i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops,details)
                    sram_cap_req,total_cp_latency,_,_,tot_latency, tot_utilization=arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops,details)
                    #print(arch.execute( i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops))
                    #print("total_cp_latency",total_cp_latency)
                    if tot_utilization>max_utilization and sram_cap_req:
                        #print(sram_cap_req,i_params, o_params, w_params, cp, cm_size, cm_type, cm_hops,details)
                        max_utilization=tot_utilization
                        best_parall=cur_gemm_parall
                        best_latency=tot_latency
                        best_cp_latency=total_cp_latency
                        best_stationary=stationary
     
    print('{:<15}, dims={}, best={}, stationary={}'.format(op['name'],dims,best_parall,best_stationary))
    result={"latency":best_latency,'utilization':max_utilization,'cp_latency':best_cp_latency}
    return result

def flashatten_mapper(config, arch: Tx8, Tx_Ty=None, details=True, Head_fused=True):  # org first para is model
    # 将Q,KV分成特定的块数，同时将不同的块分配到tile上，每个tile上一块。其中Q视为input,K&V视为权重；
    # Q:[B,Tx,H/A,A] KV=[B,Ty,H/A,A] S=[B,Tx,H/A,A]  # 这里 H_A 应该指代的总的 hidden_size
    # 外层循环次数 S/Tx,内层循环次数 S/Ty，一轮内层循环结束才输出部分和的结果S=[B,Tx,H/A,A]，而不是内层循环次数*外层循环次数
    # Head_fused 表示是否多头输入Q预加载优化
    # 外循环(沿着SoftMax方向)一次iteration会读取[N, H/A]
    #config = model.config
    dims = [config['B'], [config['S_Q'], config['S_KV']], config['H_A'], config['N_A']] # org: int( config['H_A']/config['N_A']), config['N_A']]  
    # print("config['A']",config['A'])
    # Have at least one row of data on each tile
    dims[1] = dim_norm(dims[1])
    # tx, ty 为每一份可能切分的大小
    Tx = block_range(dims[1][0], min_block=1, max_block=dims[1][0]//arch.config['TILE_NUM'])  # org: dims[1]
    Ty = block_range(dims[1][1], min_block=1, max_block=dims[1][1]//arch.config['TILE_NUM'])  # org: dims[1]
    if Tx_Ty != None:
        assert Tx_Ty[0] <= dims[1]//arch.config['TILE_NUM']
        assert Tx_Ty[1] <= dims[1]//arch.config['TILE_NUM']
        Tx, Ty = [Tx_Ty[0]], [Tx_Ty[1]]
    #print(Tx,Ty)
    head = 0
    max_utilization = 0
    best_tx_ty = []
    best_latency = 0
    best_total_cp_latency = 0
    for tx in Tx:  # outter loop iterates KV blocks
        for ty in Ty:  # inner loop iterates Q blocks
            current_tx_ty = [tx, ty]  # Br, Bc 'Q'每个数据fp16格式 每个数字占用Q//8=2字节,N_A: #head, H_A: head_dim
            Q_RoPE_wsize = config['Q']//8*tx * config['H_A']//config['N_A']/MB  # 2B* #hea d* head_dim
            K_RoPE_wsize = config['Q']//8*ty * config['H_A']//config['N_A']/MB  # iso
            if Head_fused:
                head = dims[3]
            else:
                head = 1
            i_params = [MBytes([dims[0], tx, dims[2]])+Q_RoPE_wsize, head*math.ceil(dims[1][0]//tx)]  # #head个Embedding矩阵和Q同时载入
            o_params = [MBytes([dims[0], tx, dims[2]]),head*math.ceil(dims[1][0]//tx)]  # XXX：没有考虑mask？
            w_params = [2*MBytes([dims[0], ty, dims[2]]) +K_RoPE_wsize, math.ceil(dims[1][1]//ty)]  # K+V
            # RoPE 每一行的大小  
            vector_cp_size = config['B']*tx*config['H_A']//config['N_A'] +  config['B']*ty * config['H_A']//config['N_A']  # RoPE
            flash_vector_cp_size =config['B']* 5*tx*ty  # *dims[2]
            # cp=[[2*2*tx*ty*dims[2]/G,1]]
            # cp=[[0,0],[2*2*tx*ty*dims[2]/G,1],[0,0]]
            cp = [[vector_cp_size/G, 0], [config['B']*2*2*tx*ty*dims[2]/G, 1],[flash_vector_cp_size/G, 0]]
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
        print('{:<15}, dims={}, best={}'.format('Flashatten', dims, best_tx_ty))
        if Head_fused:
            one_head_latency, one_head_cp_latency = best_latency/dims[3], best_total_cp_latency/dims[3]
        else:
            one_head_latency, one_head_cp_latency = best_latency, best_total_cp_latency
        print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format('flash_attention', one_head_latency, max_utilization ,one_head_cp_latency))
    # print(best_latency,best_total_cp_latency)
    result = {"latency": dims[3]//head*best_latency, 'utilization': max_utilization,'cp_latency': dims[3]//head*best_total_cp_latency}
    return result

def vector_mapper(op,arch: Tx8,splits=None,details=False):
    assert op['ishape']==op['oshape']
    io_shape,w_shape=op['ishape'],op['wshape']
    #assert (op['name'] in ['Modulate','RMSNorm','RMSNorm2','Hadamard','ResAdd','ResAdd2','SiLU',]) and (op['type']=='Vector')
    assert (op['type']=='Vector')
    i_split=op['ishape'][1]#RMS只能切一个维度
    if splits==None:
        if op['name'] != "RMSNorm": # in ['Hadamard','ResAdd','ResAdd2','SiLU']:
            i_split=i_split*op['ishape'][2]
        splits=block_range(i_split,min_block=1)  # 返回不超过i_split的min_block的倍数的数组
    else:
        splits = [splits]
    max_utilization = 0
    best_split = []
    best_latency =0
    total_cp_latency = 0
    #print('vector',splits)
    for split in splits:
            i_params=[MBytes(io_shape)/split,split]  # 每份大小*份数
            o_params=[MBytes(io_shape)/split,split]  # 输出切分大小和输入相同
            w_params=[MBytes(w_shape)/split,split]  # 逐点运算输出切分数等于输入切分数，默认输出切分数=输入切分数*权重切分数
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
    result = {  "latency": best_latency,
                'utilization': max_utilization, 'cp_latency': total_cp_latency} 
    return result


def PIXART_mapper(model, arch, QKV_fusion=True, preset=True, details=True):
    config = model.config
    Layers = config['L']
    spatial_config = {'B': config['B_spt'], 'S_Q': config['S_Q_spt'], 'S_KV': config['S_KV_spt'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    cross_config = {'B': config['B_cro'], 'S_Q': config['S_Q_cro'], 'S_KV': config['S_KV_cro'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    ops = model.ops
    mapping_result = {}
    '''=========================
    == Spatial Branch Mapping ==
    ========================='''
    TmTn = [256, 32] if preset else None
    mapping_result['spatial_Modulate'] = vector_mapper(ops['spatial_Modulate'],arch,splits=None,details=details)
    mapping_result['spatial_RMSNorm']= vector_mapper(ops['spatial_RMSNorm'],arch,splits=None,details=details)
    mapping_result['spatial_Q_proj'] = gemm_auto_opt_mapper(ops['spatial_Q_proj'], arch, TmTn=TmTn, details=details)
    mapping_result['spatial_K_proj'] = gemm_auto_opt_mapper(ops['spatial_K_proj'], arch, TmTn=TmTn, details=details)
    mapping_result['spatial_V_proj'] = gemm_auto_opt_mapper(ops['spatial_V_proj'], arch, TmTn=TmTn, details=details)
    Tx_Ty = [256, 256] if preset else None
    mapping_result['spatial_Flashatten'] = flashatten_mapper(spatial_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)  # FIXME
    mapping_result['spatial_ResAdd']=vector_mapper(ops['spatial_ResAdd'],arch,splits=None,details=details)
    
    '''=======================
    == Cross Branch Mapping ==
    ======================='''
    #mapping_result['spatial_RMSNorm']= vector_mapper(ops['spatial_RMSNorm'],arch,splits=None,details=details)
    mapping_result['cross_Q_proj'] =  gemm_auto_opt_mapper(ops['cross_Q_proj'], arch, TmTn=TmTn, details=details)
    mapping_result['cross_K_proj'] =  gemm_auto_opt_mapper(ops['cross_K_proj'], arch, TmTn=TmTn, details=details)
    mapping_result['cross_V_proj'] =  gemm_auto_opt_mapper(ops['cross_V_proj'], arch, TmTn=TmTn, details=details)
    Tx_Ty = [256, 256] if preset else None
    mapping_result['cross_Flashatten'] =  flashatten_mapper(cross_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)  # FIXME
    mapping_result['cross_ResAdd'] =  vector_mapper(ops['cross_ResAdd'],arch,splits=None,details=details)  
    # HACK: Gate_ResAdd *2 了, cross 无gate 这里 / 2
    
    '''=======================
    == Feed Forward Network ==
    ======================='''
    mapping_result['mlp_Modulate'] = vector_mapper(ops['mlp_Modulate'],arch,splits=None,details=details)
    mapping_result['FFNup&SiLU'] = gemm_auto_opt_mapper(ops['FFNup'],arch,TmTn=TmTn,fusion_op2=ops['SiLU'],details=details)
    TmTn = [4, 128] if preset else None
    mapping_result['FFNdown'] = gemm_auto_opt_mapper(ops['FFNdown'], arch, TmTn=TmTn, details=details)
    mapping_result['mlp_ResAdd'] = vector_mapper(ops['mlp_ResAdd'], arch, splits=None, details=details)
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
        'Total Layers', tot_latency*Layers, utilization*100, tot_cp_latency*Layers))
    return mapping_result


def STDIT2_mapper(model, arch, QKV_fusion=True, preset=True, details=True):
    config = model.config
    Layers = config['L']
    spatial_config = {'B': config['B_spt'], 'S_Q': config['S_Q_spt'], 'S_KV': config['S_KV_spt'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    temporal_config = {'B': config['B_tmp'], 'S_Q': config['S_Q_tmp'], 'S_KV': config['S_KV_tmp'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    cross_config = {'B': config['B_cro'], 'S_Q': config['S_Q_cro'], 'S_KV': config['S_KV_cro'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    ops = model.ops
    mapping_result = {}

    # 定义需要重复统计的模块及其重复次数
    repeat_modules = {
        'cross_Q_proj': 2,
        'cross_K_proj': 2,
        'cross_V_proj': 2,
        'cross_Flashatten': 2,
        'cross_Linear': 2,
        'cross_ResAdd': 2,
        'mlp_Modulate': 2,
        'FFNup&SiLU': 2,
        'FFNdown': 2,
        'mlp_ResAdd': 2,
    }

    # 映射函数
    def map_op(op_name, mapper, mapper_args=None):
        """通用映射函数"""
        if op_name not in ops:
            print(f"Warning: '{op_name}' not found in ops. Skipping mapping.")
            return

        if mapper_args is None:
            mapper_args = {}
        result = mapper(ops[op_name], arch, **mapper_args, details=details)
        mapping_result[op_name] = result

    # 执行映射
    TmTn = [256, 32] if preset else None
    Tx_Ty = [256, 256] if preset else None

    # Spatial Branch
    map_op('spatial_t2i_Modulate', vector_mapper, {'splits': None})
    map_op('spatial_RMSNorm0', vector_mapper, {'splits': None})
    map_op('spatial_RMSNorm(Q)', vector_mapper, {'splits': None})
    map_op('spatial_RMSNorm(K)', vector_mapper, {'splits': None})
    map_op('spatial_Q_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('spatial_K_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('spatial_V_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    mapping_result['spatial_Flashatten'] = flashatten_mapper(spatial_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)
    map_op('spatial_Linear', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('spatial_ResAdd', vector_mapper, {'splits': None})

    # Temporal Branch
    map_op('temporal_t2i_Modulate', vector_mapper, {'splits': None})
    map_op('temporal_RMSNorm0', vector_mapper, {'splits': None})
    map_op('temporal_RMSNorm(Q)', vector_mapper, {'splits': None})
    map_op('temporal_RMSNorm(K)', vector_mapper, {'splits': None})
    map_op('temporal_Q_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_K_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_V_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_QK^T', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_Softmax', vector_mapper, {})
    map_op('temporal_AV', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_Linear', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('temporal_ResAdd', vector_mapper, {'splits': None})

    # Cross Branch
    map_op('cross_Q_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('cross_K_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('cross_V_proj', gemm_auto_opt_mapper, {'TmTn': TmTn})
    mapping_result['cross_Flashatten'] = flashatten_mapper(cross_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)
    map_op('cross_Linear', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('cross_ResAdd', vector_mapper, {'splits': None})

    # Feed Forward Network
    map_op('mlp_t2i_Modulate', vector_mapper, {'splits': None})
    map_op('mlp_RMSNorm0', vector_mapper, {'splits': None})
    map_op('FFNup', gemm_auto_opt_mapper, {'TmTn': TmTn, 'fusion_op2': ops['SiLU']})
    TmTn = [4, 128] if preset else None
    map_op('FFNdown', gemm_auto_opt_mapper, {'TmTn': TmTn})
    map_op('mlp_ResAdd', vector_mapper, {'splits': None})

    # 统计结果
    print('-' * 40 + 'mapping_result' + '-' * 40)
    tot_latency = 0
    tot_cp_latency = 0
    tot_utilization = 0

    for key, item in mapping_result.items():
        try:
            # 获取重复次数，默认为 1
            repeat = repeat_modules.get(key, 1)
            tot_latency += item['latency'] * repeat
            tot_cp_latency += item['cp_latency'] * repeat
            tot_utilization += item['utilization'] * repeat
            print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(
                key, item['latency'], item['utilization'] * 100, item['cp_latency']))
        except:
            print('{:<15}, No suitable mapping result! '.format(key))

    utilization = tot_cp_latency / (tot_latency + 1e-35)
    mapping_result['Total'] = {
        "latency": tot_latency, 'utilization': utilization, 'cp_latency': tot_cp_latency}
    print('{:<15}, latency(ms)={:>10.6f}, utilization(%)={:>10.6f}, compute latency(ms)={:>10.6f}'.format(
        'Total Layers', tot_latency * Layers, utilization * 100, tot_cp_latency * Layers))

    return mapping_result


def manual_mapper(model, arch, QKV_fusion=True, preset=True, details=True):
    # 指定映射
    Layers = model.config['L']
    ops = model.ops
    
    mapping_result = {}
    if details:
        print('-'*40+'mapping_processing'+'-'*40)
    #1
    #mapping_result['RMSNorm']=vector_mapper(ops['RMSNorm'],arch,splits=None,details=details)
    
    if QKV_fusion:
        mapping_result['RMSNorm']=vector_mapper(ops['RMSNorm'],arch,splits=None,details=details)
        ops["QKV_fusion"] = model.gen_gemm("QKV_fusion", [config["B"], config["S"], config["D_QKV"], 3*config["H_QKV"]])
        TmTn = [256, 8] if preset else None
        #mapping_result['QKV_fusion']=gemm_auto_opt_mapper(ops['QKV_fusion'],arch,TmTn=TmTn,fusion_op1=None,details=details)
        mapping_result['QKV_fusion'] = gemm_auto_opt_mapper(ops['QKV_fusion'], arch, TmTn=TmTn,details=details)
        del ops['Q_proj']
        del ops['K_proj']
        del ops['V_proj']
        del ops['RMSNorm']
    else:
        TmTn = [256, 32] if preset else None
        mapping_result['RMSNorm']= vector_mapper(ops['RMSNorm'],arch,splits=None,details=details)
        mapping_result['Q_proj'] = gemm_auto_opt_mapper(ops['Q_proj'], arch, TmTn=TmTn, details=details)
        mapping_result['K_proj'] = gemm_auto_opt_mapper(ops['K_proj'], arch, TmTn=TmTn, details=details)
        mapping_result['V_proj'] = gemm_auto_opt_mapper(ops['V_proj'], arch, TmTn=TmTn, details=details)
        del ops['RMSNorm']
        del ops['Q_proj']
    
    # 2
    
    Tx_Ty = [256, 256] if preset else None  # wanghuizheng
    mapping_result['Flashatten'] = flashatten_mapper(model, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)
    del ops['RoPE(Q)']
    del ops['RoPE(K)']
    del ops['QK^T']
    del ops['Softmax']
    del ops['AV']
    
    mapping_result['Linear']=gemm_auto_opt_mapper(ops['Linear'],arch,details=details)

    mapping_result['RMSNorm2']=vector_mapper(ops['RMSNorm2'],arch,splits=None,details=details)
    mapping_result['ResAdd']=vector_mapper(ops['ResAdd'],arch,splits=None,details=details)
    #3
    TmTn=None# [16,256] #if preset else None
    #mapping_result['FFNup']=gemm_auto_opt_mapper(ops['FFNup'],arch,TmTn=TmTn,details=details)
    #mapping_result['SiLU']=vector_mapper(ops['SiLU'],arch,splits=None,details=details)
    mapping_result['FFNup&SiLU']=gemm_auto_opt_mapper(ops['FFNup'],arch,TmTn=TmTn,fusion_op2=ops['SiLU'],details=details)
    del ops['SiLU']
    mapping_result['FFNgate'] = gemm_auto_opt_mapper(ops['FFNgate'], arch, TmTn=TmTn, details=details)
    mapping_result['Hadamard'] = vector_mapper(ops['Hadamard'], arch, splits=None)
    TmTn = [4, 128] if preset else None
    mapping_result['FFNdown'] = gemm_auto_opt_mapper(ops['FFNdown'], arch, TmTn=TmTn, details=details)
    mapping_result['ResAdd2'] = vector_mapper(ops['ResAdd2'], arch, splits=None, details=details)
    
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
        'Total Layers', tot_latency*Layers, utilization*100, tot_cp_latency*Layers))
    return mapping_result


if __name__ == "__main__":
    import os

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 切换工作目录到当前文件所在的目录
    os.chdir(current_dir)
    import pprint
    #config_path = "./input/transformer/megatron_51_1280_720.json"
    config_path = "./input/transformer/ds_204_640_360.json"
    filename = os.path.basename(config_path)  # 获取文件名
    sora_config =load_config(config_path)
    llama7b = tbk.STDIT2_block(sora_config, is_sp=filename.startswith("ds"))
    pprint.pprint(llama7b.config)
    tx8_config = load_config('hardware_parameter.json')
    hardware = Tx8(tx8_config)
    pprint.pprint(hardware.config)
    config = llama7b.config
    temporal_config = {'B': config['B_tmp'], 'S_Q': config['S_Q_tmp'], 'S_KV': config['S_KV_tmp'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
    mapping_result = flashatten_mapper(temporal_config, hardware, Tx_Ty=None, details=True, Head_fused=True)
    # preset 是否使用预设切分;details是否打印映射的详细信息
    #import ipdb; ipdb.set_trace()
    mapping_result = STDIT2_mapper(llama7b, hardware, preset=False, details=True)
