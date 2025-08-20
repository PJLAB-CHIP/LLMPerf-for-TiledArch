
import json
import pandas as pd
import math
from typing import Union
TB=1024**4
GB=1024**3
MB=1024**2
KB=1024
T=1000**4
G=1000**3
M=1000**2
K=1000
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
# def MBytes(list0,bytes=2):
#     if list0==None or list0==0:
#         return 0
#     #List 维度乘积
#     res=1 
#     for i in list0:
#         res*=i
#     return res/1024/1024*bytes

def MBytes(dims, bytes_per_element=2):
    """Calculates the size of a multi-dimensional array in MB.

    Args:
        dims: A list or tuple representing the dimensions of the array.
        bytes_per_element: The size of each element in bytes (default is 2).

    Returns:
        The size of the array in MB. Returns 0 if dims is None, empty, or contains non-positive values.  Raises TypeError if dims is not a list or tuple.
    """
    if dims is None:
        return 0  # Explicitly handle None case

    if not isinstance(dims, (list, tuple)):
        raise TypeError("dims must be a list or tuple")
    if not dims or any(dim <= 0 for dim in dims):
        return 0  # Handle empty or invalid dimensions

    total_elements = math.prod(dims) # Efficiently calculates the product of all elements in the list.
    return total_elements * bytes_per_element / (1024 ** 2)

# def min_multiple_of(num,factor=16):
#     return math.ceil(num/factor)*factor
# def dim_norm(dims,tile_num=16):
#     #将维度规则到tile_num的倍数
#     newdims=[]
#     for dim in dims:
#         newdims.append(min_multiple_of(dim,tile_num))
#     return newdims

def dim_norm(dims, tile_num=16):
    """Rounds dimensions up to the nearest multiple of tile_num.

    Args:
        dims: A list or tuple of dimensions.
        tile_num: The target multiple.

    Returns:
        A list of dimensions rounded up to the nearest multiple of tile_num.
    """
    return [ (dim + tile_num - 1) // tile_num * tile_num for dim in dims ]

# def dim_analysis(optype,dims,para_dims):
#     #将gemm矩阵维度按并行切分度重新计算数据shape
#     if optype=='GEMM':#[b,m,k,n]
#         reduce=True if para_dims[2]>1 else False
#         newdims=[math.ceil(dims[0]/para_dims[0]),math.ceil(dims[1]/para_dims[1]),math.ceil(dims[2]/para_dims[2]),math.ceil(dims[3]/para_dims[3])]
#         i_shape=[math.ceil(newdims[0]),math.ceil(newdims[1]),math.ceil(newdims[2])]
#         w_shape=[math.ceil(newdims[2]),math.ceil(newdims[3])]
#         o_shape=[math.ceil(newdims[0]),math.ceil(newdims[1]),math.ceil(newdims[3])]
#         return newdims,i_shape,o_shape,w_shape,reduce

def dim_analysis(optype, dims, para_dims):
    """Analyzes dimensions for GEMM operations with parallel partitioning.

    Args:
        optype: The operation type ('GEMM').
        dims: A list or tuple representing the original dimensions [b, m, k, n].
        para_dims: A list or tuple representing the parallel partitioning dimensions [b_partitions, m_partitions, k_partitions, n_partitions].

    Returns:
        A tuple containing:
            - newdims: The new dimensions after partitioning [b_new, m_new, k_new, n_new].
            - i_shape: The shape of the input tensor [b_new, m_new, k_new].
            - w_shape: The shape of the weight tensor [k_new, n_new].
            - o_shape: The shape of the output tensor [b_new, m_new, n_new].
            - reduce: A boolean indicating whether reduction is needed (True if k_partitions > 1, False otherwise).

        Returns None if optype is not 'GEMM' or if input dimensions are invalid.

    """
    if optype != 'GEMM':
        return None
    if not all(isinstance(x, (list, tuple)) and len(x) == 4 for x in [dims, para_dims]):
        return None
    if any(dim <= 0 for dim in dims + para_dims):
        return None

    newdims = [(dim + p -1 ) // p for dim, p in zip(dims, para_dims)] #Efficiently calculates new dimensions
    reduce = para_dims[2] > 1

    return  newdims, \
            [newdims[0], newdims[1], newdims[2]], \
            [newdims[2], newdims[3]], [newdims[0], \
            newdims[1], newdims[3]], \
            reduce

# def block_range(dim,min_block=1,max_block=None):
#     #遍历dim可以因式分解的所有公因子，满足大于等于min_block，且为min_block的倍数，且小于等于max_block
#     if max_block==None:
#         max_block=dim
#     factors = []
#     sqrt_n = int(math.sqrt(dim))
#     for i in range(1, sqrt_n + 1):  # i should > min_block
#         if dim % i == 0 :
#             if i % min_block ==0 and i <= max_block:
#                 factors.append(i)
#             if i != dim // i:
#                 if dim // i % min_block ==0 and dim // i <= max_block:
#                     factors.append(dim // i)
#     return factors
def block_range(dim, min_block=1, max_block=None):
    """
    Iterates through all factors of dim that are multiples of min_block and less than or equal to max_block.

    Args:
        dim: The integer to find factors of.
        min_block: The minimum value of a factor.  Factors must be multiples of this value.
        max_block: The maximum value of a factor. Defaults to dim.

    Returns:
        A list of factors that satisfy the conditions.  The list is sorted in ascending order.
    """
    if max_block is None:
        max_block = dim
    elif max_block == 0:
        max_block = 1

    factors = []
    for i in range(min_block, max_block + 1, min_block):  #Efficiently iterate through multiples of min_block
        if dim % i == 0:
            factors.append(i)

    factors.sort() #Ensure ascending order for consistency
    return factors
if __name__ == "__main__":
    '''
    new_dims=dim_norm([16,4096,5,511],factor=16)
    print(new_dims)
    for dim in new_dims:
        print(block_range(dim,min_block=16))
    print(dim_analysis('GEMM',new_dims,[1,128,64,256]))
    '''
    dims = dim_norm([32,128,4096,11008,12288],tile_num = 16*64)
    print(dims)
    print(block_range(dims[0]))
    print(block_range(dims[1]))
    print(block_range(dims[2]))
    print(block_range(dims[3]))
    print(block_range(dims[4]))
