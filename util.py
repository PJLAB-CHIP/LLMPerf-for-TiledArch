
import json
import pandas as pd
import math
from typing import Union
def load_config(input_path):
    # 读取json配置文件
    with open(input_path, 'r') as file:
        # 从文件中加载 JSON 数据
        config = json.load(file)
    return config
def save_file(data, file_path):
    # 转换为list+dict类型
    df = pd.DataFrame(data)
    # 将DataFrame保存到Excel中，index参数用于指定是否包含行索引
    df.to_excel(file_path, index=False)
def MBytes(list0,bytes=2):
    "List 维度乘积"
    res=1 
    for i in list0:
        res*=i
    return res/1024/1024*bytes

def dim_analysis(optype,dims,para_dims):
    #print(dims)
    #print(para_dims)
    if optype=='GEMM':#[b,m,k,n]
        reduce=True if para_dims[2]>1 else False
        new_shape=[dims[0]/para_dims[0],dims[1]/para_dims[1],dims[2]/para_dims[2],dims[3]/para_dims[3]]
        i_shape=[math.ceil(new_shape[0]),math.ceil(new_shape[1]),math.ceil(new_shape[2])]
        w_shape=[math.ceil(new_shape[2]),math.ceil(new_shape[3])]
        o_shape=[math.ceil(new_shape[0]),math.ceil(new_shape[1]),math.ceil(new_shape[3])]
        return new_shape,i_shape,o_shape,w_shape,reduce

def dim_split(dim,split:Union[None,int]=None):
    alter_groups=[]
    if split!=None:
        alter_groups.append((split))
    else:
        max_split=math.ceil(math.log2(dim)+1)
        for i in range(max_split):
            temp=2**i
            alter_groups.append(temp)
    return alter_groups #划分的数量