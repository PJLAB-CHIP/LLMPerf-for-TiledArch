import json
import pandas as pd
import ipdb
from copy import deepcopy
import math

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

class Modulate():
    def __init__(self, config: dict, name: str, is_sp: bool) -> None:
        self.config = config
        self.name = name
        self.ops = {}
        self.factor = 16 if is_sp else 1
        self.construct_model()
        
    def construct_model(self):
        GB = 2**30
        Modulate_input_shape = [self.config['B'], math.ceil(self.config["S_Q"] / self.factor), self.config['D_QKV']]
        Modulate_weight_shape = [1, self.config['D_QKV']] #[self.config['D_O'], self.config['H_O']]
        Modulate_output_shape = Modulate_input_shape
        RMSNorm_compute = 4*Modulate_input_shape[0]*Modulate_input_shape[1]*Modulate_input_shape[2]/GB 
        Modulate_compute = 2*Modulate_input_shape[0]*Modulate_input_shape[1]*Modulate_input_shape[2]/GB  # Norm need 4BSH ops if rsqrt is viewed as one op else 4BSH, then *(1+scale)+shift
        self.ops[self.name+"_"+"RMSNorm0"] = {"name": "RMSNorm0", "type": "Vector", "ishape": Modulate_input_shape, "wshape": Modulate_weight_shape, "oshape": Modulate_output_shape, "compute": RMSNorm_compute}
        self.ops[self.name+"_"+"t2i_Modulate"] = {"name": "Modulate", "type": "Vector", "ishape": Modulate_input_shape, "wshape": Modulate_weight_shape, "oshape": Modulate_output_shape, "compute": Modulate_compute}

class Gate_ResAdd():
    '''
    Construct each op after MHSA on the config file
    '''
    def __init__(self, config: dict, name: str, is_sp: bool) -> None:
        self.config = config
        self.name = name
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.factor = 16 if is_sp else 1
        self.construct_model()
        
    def construct_model(self):
        GB = 2**30
        ResAdd_input_shape = [self.config['B'], math.ceil(self.config["S_Q"] / self.factor), self.config['D_QKV']]
        ResAdd_weight_shape = [1, self.config['D_QKV']]#[self.config["D_O"], self.config["H_O"]]
        ResAdd_output_shape = ResAdd_input_shape
        ResAdd_compute = 2*ResAdd_input_shape[0]*ResAdd_input_shape[1]*ResAdd_input_shape[2]/GB
        self.ops[self.name+"_"+"ResAdd"] = {"name":"ResAdd", "type": "Vector", "ishape": ResAdd_input_shape, "wshape": ResAdd_weight_shape, "oshape": ResAdd_output_shape, "compute": ResAdd_compute}

class QKV_Proj():
    '''
    Construct each op in QKV projection on the config file
    '''
    def __init__(self, config: dict, name: str) -> None:
        self.config = config
        self.name = name
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.construct_model()
        
    def construct_model(self):
        GB = 2**30
        Proj_input_shape = [self.config['B'], math.ceil(self.config["S_Q"] / self.factor), self.config['D_QKV']]
        Proj_weight_shape = [self.config['D_QKV'], self.config['H_QKV']]
        Proj_output_shape = [Proj_input_shape[0], Proj_input_shape[1], Proj_weight_shape[1]]
        Proj_compute = 2*Proj_input_shape[0]*Proj_input_shape[1]*Proj_weight_shape[0]*Proj_weight_shape[1]/GB
        self.ops[self.name+"_"+"QKVProj"] = {"name":"QKVProj", "type": "GEMM", "ishape": Proj_input_shape, "wshape": Proj_weight_shape, "oshape": Proj_output_shape, "compute": Proj_compute}
        
class MHSA_block():
    '''
    Construct each op in Multi-Head Self Attention on the config file
    op = {name: {type:"", ishape:[B, X, Y], wshape:[X, Y]/None, oshape:[B, X, Y], compute:""}}
    input: config, tmp_config, cross_config
    '''
    def __init__(self, config, name: str, is_sp: bool) -> None:
        self.config = config
        self.name = name
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.factor = 16 if is_sp else 1
        self.construct_model()
        
    def construct_model(self):
        G = 2 ** 30
        # 1. Q_proj phase
        Q_Proj_input_shape =  [self.config["B"], math.ceil(self.config["S_Q"] / self.factor), self.config["D_QKV"]]
        # weight_shape: (hidd_size, hidd_size*3) NOTE: QKV Projections are combined
        Proj_weight_shape = [self.config["D_QKV"], self.config["H_QKV"] // 3]
        Q_Proj_output_shape = [Q_Proj_input_shape[0], Q_Proj_input_shape[1], Proj_weight_shape[1]]
        Q_Proj_compute = 2*Q_Proj_input_shape[0]*Q_Proj_input_shape[1]*Proj_weight_shape[0]*Proj_weight_shape[1] / G
        self.ops[self.name+"_"+"Q_proj"] = {"name":"Q_proj",
                                "type": "GEMM", 
                                "ishape": Q_Proj_input_shape, 
                                "wshape": Proj_weight_shape, 
                                "oshape": Q_Proj_output_shape, 
                                "compute":Q_Proj_compute}

        # 2. K_proj
        KV_Proj_input_shape =  [self.config["B"], math.ceil(self.config["S_KV"] / self.factor), self.config["D_QKV"]]
        KV_Proj_output_shape = [KV_Proj_input_shape[0], KV_Proj_input_shape[1], Proj_weight_shape[1]]
        KV_Proj_compute = 2*KV_Proj_input_shape[0]*KV_Proj_input_shape[1]*Proj_weight_shape[0]*Proj_weight_shape[1]  /G
        self.ops[self.name+"_"+"K_proj"] = {"name":"K_proj",
                                            "type": "GEMM", 
                                            "ishape": KV_Proj_input_shape, 
                                            "wshape": Proj_weight_shape, 
                                            "oshape": KV_Proj_output_shape, 
                                            "compute":KV_Proj_compute}
        # 3. V_proj
        self.ops[self.name+"_"+"V_proj"] = {"name":"V_proj", 
                                            "type": "GEMM", 
                                            "ishape": KV_Proj_input_shape, 
                                            "wshape": Proj_weight_shape, 
                                            "oshape": KV_Proj_output_shape, 
                                            "compute":KV_Proj_compute}
        # 4. RMSNorm(Q,K)
        RMSNorm_input_shape = [self.config["B"]*self.config["N_A"]*self.factor, math.ceil(self.config["S_Q"] / self.factor), self.config["H_A"]]
        RMSNorm_weight_shape = None  # [1, self.config["D_QKV"]]
        RMSNorm_output_shape = [self.config["B"]*self.config["N_A"]*self.factor, math.ceil(self.config["S_Q"] / self.factor), self.config["H_A"]]
        if self.config["S_Q"] == self.config["S_KV"]:
            # compute(GFLOPS) = 4*batch_size*input_row*input_col/1024/1024/1024
            RMSNorm_compute = 4 * RMSNorm_input_shape[0] * RMSNorm_input_shape[1] * RMSNorm_input_shape[2] / G
            self.ops[self.name+"_"+"RMSNorm(Q)"] = {"name":"RMSNorm(Q)", 
                                "type":  "Vector", 
                                "ishape": RMSNorm_input_shape, 
                                "wshape": RMSNorm_weight_shape, 
                                "oshape": RMSNorm_output_shape, 
                                "compute":RMSNorm_compute}
            self.ops[self.name+"_"+"RMSNorm(K)"] = {"name":"RMSNorm(K)", 
                                "type":  "Vector", 
                                "ishape": RMSNorm_input_shape, 
                                "wshape": RMSNorm_weight_shape, 
                                "oshape": RMSNorm_output_shape, 
                                "compute":RMSNorm_compute}
        # cross attention doesn't need RoPE
        # 5. RoPE(Q) only for each head
        RoPE_input_shape = [self.config["B"]*self.config["N_A"]*self.factor, math.ceil(self.config["S_Q"] / self.factor), self.config["H_A"]]
        # split col into each head
        # RoPE_input_shape[2] = int(RoPE_input_shape[2]/self.config["N_A"])
        RoPE_weight_shape = [2*RoPE_input_shape[1], RoPE_input_shape[2]]
        RoPE_output_shape = RoPE_input_shape
        RoPE_compute = 4*RoPE_input_shape[0]*RoPE_input_shape[1]*RoPE_input_shape[2] / G
        if self.config["S_Q"] == self.config["S_KV"]:  # Not cross_attn
            self.ops[self.name+"_"+"RoPE(Q)"] = {"name":"RoPE(Q)",
                                                "type": "Vector", 
                                                "ishape":RoPE_input_shape, 
                                                "wshape": RoPE_weight_shape, 
                                                "oshape": RoPE_output_shape, 
                                                "compute": RoPE_compute}
            # 6. RoPE(K) only for each head
            self.ops[self.name+"_"+"RoPE(K)"] = {"name":"RoPE(K)", 
                                                "type": "Vector", 
                                                "ishape": RoPE_input_shape, 
                                                "wshape": RoPE_weight_shape, 
                                                "oshape": RoPE_output_shape, 
                                                "compute": RoPE_compute}
        # 7. QK^{T}
        QK_input_shape = [self.config["B"]*self.config["N_A"], self.config["S_Q"], self.config["H_A"]]  # BA N C 
        QK_weight_shape = [self.config["H_A"], self.config['S_KV']]
        QK_output_shape = [QK_input_shape[0], QK_input_shape[1], QK_weight_shape[1]]
        QK_compute = 2*math.prod(QK_input_shape) * QK_weight_shape[1] / G
        self.ops[self.name+"_"+"QK^T"] = {"name":"QK^T", 
                                        "type": "GEMM", 
                                        "ishape": QK_input_shape, 
                                        "wshape": QK_weight_shape, 
                                        "oshape": QK_output_shape, 
                                        "compute":QK_compute}
        # 8. Softmax
        Softmax_input_shape = deepcopy(QK_output_shape)
        Softmax_weight_shape = None
        Softmax_output_shape = Softmax_input_shape
        Softmax_compute = 5*Softmax_input_shape[0]*Softmax_input_shape[1]*Softmax_input_shape[2]/G
        self.ops[self.name+"_"+"Softmax"] = {"name":"Softmax", 
                                            "type": "Vector", 
                                            "ishape": Softmax_input_shape, "wshape": Softmax_weight_shape, "oshape": Softmax_output_shape, "compute": Softmax_compute}
        # 9. AV
        AV_input_shape = deepcopy(Softmax_output_shape)
        AV_weight_shape = [self.config['S_KV'], self.config["H_A"]]
        AV_output_shape = [AV_input_shape[0], AV_input_shape[1], AV_weight_shape[1]]
        AV_compute = 2*math.prod(AV_input_shape)*AV_weight_shape[1]/G
        self.ops[self.name+"_"+"AV"] = {"name":"AV", 
                                        "type": "GEMM", 
                                        "ishape":AV_input_shape, 
                                        "wshape": AV_weight_shape, 
                                        "oshape": AV_output_shape, 
                                        "compute": AV_compute}
        # 10. Linear 
        Linear_input_shape = [self.config['B'], math.ceil(self.config["S_Q"] / self.factor), self.config["D_O"]]
        Linear_weight_shape = [self.config["D_O"], self.config["H_O"]]
        Linear_output_shape = Linear_input_shape
        Linear_compute = 2*Linear_input_shape[0]*Linear_input_shape[1]*Linear_weight_shape[0]*Linear_weight_shape[1]/G
        self.ops[self.name+"_"+"Linear"] = {"name":"Linear", 
                                            "type": "GEMM", 
                                            "ishape": Linear_input_shape, 
                                            "wshape": Linear_weight_shape, 
                                            "oshape": Linear_output_shape, 
                                            "compute": Linear_compute}

class FFN_block():
    '''
    Construct each op in Feed Forward Network
    '''      
    def __init__(self, config, is_sp: bool) -> None:
        self.config = config
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.factor = 16 if is_sp else 1
        self.construct_model()
        
    def construct_model(self):
        GB = 2**30
        # 12. RMSNorm
        RMSNorm_input_shape = [self.config["B"], math.ceil(self.config["S_Q"] / self.factor), self.config["D_QKV"]]
        RMSNorm_weight_shape = [1, self.config["D_QKV"]]
        RMSNorm_output_shape = RMSNorm_input_shape
        RMSNorm_compute = 4 * RMSNorm_input_shape[0] * RMSNorm_input_shape[1] * RMSNorm_input_shape[2] / GB
        self.ops["RMSNorm2"] = {"name":"RMSNorm2", 
                                "type": "Vector", 
                                "ishape":RMSNorm_input_shape, 
                                "wshape": RMSNorm_weight_shape, 
                                "oshape":RMSNorm_output_shape, 
                                "compute":RMSNorm_compute}
        # 13. FFNup
        FFNup_input_shape = [self.config["B"], math.ceil(self.config["S_Q"] / self.factor), self.config["D_FU"]]
        FFNup_weight_shape = [self.config['D_FU'], self.config["H_FU"]]

        FFNup_output_shape = [FFNup_input_shape[0], FFNup_input_shape[1], FFNup_weight_shape[1]]
        FFNup_compute = 2*FFNup_input_shape[0]*FFNup_input_shape[1]*FFNup_weight_shape[0]*FFNup_weight_shape[1]/GB
        self.ops["FFNup"] = {"name":"FFNup", 
                            "type": "GEMM", 
                            "ishape":FFNup_input_shape, 
                            "wshape": FFNup_weight_shape, 
                            "oshape": FFNup_output_shape, 
                            "compute": FFNup_compute}
        # 14. FFNgate
        # self.ops["FFNgate"] =   {"name":"FFNgate", 
        #                         "type": "GEMM", 
        #                         "ishape":FFNup_input_shape, 
        #                         "wshape": FFNup_weight_shape, 
        #                         "oshape": FFNup_output_shape, 
        #                         "compute": FFNup_compute}
        # 15. SiLU
        SiLU_input_shape = deepcopy(FFNup_output_shape)
        SiLU_weight_shape = None
        SiLU_output_shape = SiLU_input_shape
        SiLU_compute = 7*SiLU_input_shape[0]*SiLU_input_shape[1]*SiLU_input_shape[2]/GB
        self.ops["SiLU"] = {"name":"SiLU", 
                            "type": "Vector", 
                            "ishape": SiLU_input_shape, 
                            "wshape": SiLU_weight_shape, 
                            "oshape": SiLU_output_shape, 
                            "compute": SiLU_compute}
        # 16. Hadamard
        # Hadamard_input_shape = deepcopy(SiLU_output_shape)
        # Hadamard_weight_shape = [FFNup_output_shape[1], FFNup_output_shape[2]]
        # Hadamard_output_shape = Hadamard_input_shape
        # Hadamard_compute = Hadamard_input_shape[0]*Hadamard_input_shape[1]*Hadamard_input_shape[2]/GB 
        # self.ops["Hadamard"] = {"name":"Hadamard", 
        #                         "type": "Vector", 
        #                         "ishape": Hadamard_input_shape, 
        #                         "wshape": Hadamard_weight_shape, 
        #                         "oshape": Hadamard_output_shape, 
        #                         "compute": Hadamard_compute}
        # 17. FFNdown
        FFNdown_input_shape = deepcopy(SiLU_output_shape)#deepcopy(Hadamard_output_shape)
        FFNdown_weight_shape = [self.config["D_FD"], self.config["H_FD"]]
        FFNdown_output_shape = [FFNdown_input_shape[0], FFNdown_input_shape[1], FFNdown_weight_shape[1]]
        FFNdown_compute = 2*FFNdown_input_shape[0]*FFNdown_input_shape[1]*FFNdown_weight_shape[0]*FFNdown_weight_shape[1]/GB
        self.ops["FFNdown"] = {"name":"FFNdown", 
                                "type": "GEMM", 
                                "ishape":FFNdown_input_shape, 
                                "wshape": FFNdown_weight_shape, 
                                "oshape": FFNdown_output_shape, 
                                "compute": FFNdown_compute}

class PIXART_block():
    def __init__(self, config) -> None:
        self.config = config
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        #self.cro
        self.construct_model()
    
    def construct_model(self):
        spatial_config = {"B": self.config["B_spt"], "S_Q": self.config["S_Q_spt"], "S_KV": self.config["S_KV_spt"], "D_QKV": self.config["D_QKV"], 
                        "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_spt"], "H_O": self.config["H_O_spt"] }
        cross_config = {"B": self.config["B_cro"], "S_Q": self.config["S_Q_cro"], "S_KV": self.config["S_KV_cro"], "D_QKV": self.config["D_QKV"], 
                        "H_QKV": self.config["H_QKV"],"N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_cro"], "H_O": self.config["H_O_cro"],
                        "D_FU": self.config["D_FU"], "H_FU": self.config["H_FU"], "D_FD": self.config["D_FD"], "H_FD": self.config["H_FD"]}
        
        self.spatial_modulate = Modulate(spatial_config, name="spatial")
        self.spatial_block = MHSA_block(spatial_config, name="spatial")
        self.spatial_gate_resadd = Gate_ResAdd(spatial_config, name="spatial")
        self.cross_block = MHSA_block(cross_config, name="cross")
        self.cross_gate_resadd = Gate_ResAdd(cross_config, name="cross")
        self.mlp_modulate = Modulate(cross_config, name="mlp")
        self.ffn_block = FFN_block(cross_config)
        self.mlp_gate_resadd = Gate_ResAdd(cross_config, name="mlp")
        op_list = [self.spatial_modulate.ops, self.spatial_block.ops, self.spatial_gate_resadd.ops, 
                self.cross_block.ops, self.cross_gate_resadd.ops, self.mlp_modulate.ops, self.ffn_block.ops, self.mlp_gate_resadd.ops]
        for op_dict in op_list:
            self.ops.update(op_dict)
        print(self.ops.keys())
        
        
class STDIT2_block():
    def __init__(self, config, is_sp) -> None:
        self.config = config
        self.spatial_config= None
        self.temporal_config = None
        self.cross_config = None
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.sp = is_sp
        self.construct_model()

        
    def construct_model(self):
        spatial_config = {"B": self.config["B_spt"], "S_Q": self.config["S_Q_spt"], "S_KV": self.config["S_KV_spt"], "D_QKV": self.config["D_QKV"], 
                        "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_spt"], "H_O": self.config["H_O_spt"] }
        temporal_config = {"B": self.config["B_tmp"], "S_Q": self.config["S_Q_tmp"], "S_KV": self.config["S_KV_tmp"], "D_QKV": self.config["D_QKV"], 
                        "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_tmp"], "H_O": self.config["H_O_tmp"] }
        cross_config = {"B": self.config["B_cro"], "S_Q": self.config["S_Q_cro"], "S_KV": self.config["S_KV_cro"], "D_QKV": self.config["D_QKV"], 
                        "H_QKV": self.config["H_QKV"],"N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_cro"], "H_O": self.config["H_O_cro"],
                        "D_FU": self.config["D_FU"], "H_FU": self.config["H_FU"], "D_FD": self.config["D_FD"], "H_FD": self.config["H_FD"]}
        self.spatial_config, self.temporal_config, self.cross_config= spatial_config, temporal_config, cross_config
        self.spatial_modulate = Modulate(spatial_config, name="spatial", is_sp=self.sp)
        self.spatial_block = MHSA_block(spatial_config, name="spatial", is_sp=self.sp)
        self.spatial_gate_resadd = Gate_ResAdd(spatial_config, name="spatial", is_sp=self.sp)
        self.temporal_modulate = Modulate(temporal_config, name="temporal", is_sp=self.sp)
        self.temporal_block = MHSA_block(temporal_config, name="temporal", is_sp=self.sp)
        self.temporal_gate_resadd = Gate_ResAdd(temporal_config, name="temporal", is_sp=self.sp)
        self.cross_block = MHSA_block(cross_config, name="cross", is_sp=self.sp)
        self.cross_gate_resadd = Gate_ResAdd(cross_config, name="cross", is_sp=self.sp)
        self.mlp_modulate = Modulate(cross_config, name="mlp", is_sp=self.sp)
        self.ffn_block = FFN_block(cross_config, is_sp=self.sp)
        self.mlp_gate_resadd = Gate_ResAdd(cross_config, name="mlp", is_sp=self.sp)
        op_list = [self.spatial_modulate.ops, self.spatial_block.ops, self.spatial_gate_resadd.ops, 
                    self.temporal_modulate.ops, self.temporal_block.ops, self.temporal_gate_resadd.ops, 
                    self.cross_block.ops, self.cross_gate_resadd.ops, self.mlp_modulate.ops, self.ffn_block.ops, self.mlp_gate_resadd.ops]
        for op_dict in op_list:
            self.ops.update(op_dict)
        print(self.ops.keys())
            
        
        
class Llama_block():
    '''
    Construct each op in Llama block based on the config file
    op = {name: {type:"", ishape:[B, X, Y], wshape:[X, Y]/None, oshape:[B, X, Y], compute:""}}
    input: config (B, S, H, A, L, H', Q) == (batch size, sequence len, hidden size, head num, hidden layer, H', quantization)
    '''
    def __init__(self, config) -> None:
        self.config = config
        # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
        self.ops = {}
        self.construct_model()

    def cal_in_out_size(self, in_out_shape):
        '''
        given input/output shape(batch_size, row, col), return size(MB)
        formula: batch_size * shape_row * shape_col * quant / 8 / 1024 / 1024
        '''
        return in_out_shape[0] * in_out_shape[1] * in_out_shape[2] * self.config["Q"] / 8 / 1024 / 1024
    def cal_weight_size(self, weight_shape):
        '''
        given weight shape:(row, col), return size(MB)
        formula: shape_row * shape_col * quant / 8 / 1024 / 1024
        '''
        return weight_shape[0] * weight_shape[1] * self.config["Q"] / 8 / 1024 / 1024
    def print_ops(self):
        for name, ops in self.ops.items():
            res = ""
            for key, val in ops.items():
                res += f"{val} -- "
            print(res)
    def save_ops(self, data_path):
        data = []
        for name, ops in self.ops.items():
            res = ops
            data.append(res)
        save_file(data, data_path)
    def gen_gemm(self,name,dims):
        assert len(dims)==4
        unit = 1000 * 1000 * 1000
        ishape=[dims[0],dims[1],dims[2]]
        wshape = [dims[2],dims[3]]
        oshape = [dims[0],dims[1],dims[3]]
        Proj_compute = 2*ishape[0]*ishape[1]*wshape[0]*wshape[1]/unit
        return {"name":name,"type": "GEMM", "ishape":ishape, "wshape": wshape, "oshape":oshape, "compute":Proj_compute}
    
    def construct_model(self):
        #ipdb.set_trace()
        # GFLOPS unit
        unit = 1000 * 1000 * 1000
        #unit = 1024 * 1024 * 1024
        # 1. RMSNorm phase
        # (batch_size, row, col)
        RMSNorm_input_shape = [self.config["B"], self.config["S"], self.config["D_QKV"]]
        RMSNorm_weight_shape = [1, self.config["D_QKV"]]
        RMSNorm_output_shape = [self.config["B"], self.config["S"],self.config["D_QKV"]]
        # compute(GFLOPS) = 4*batch_size*input_row*input_col/1024/1024/1024
        RMSNorm_compute = 4 * RMSNorm_input_shape[0] * RMSNorm_input_shape[1] * RMSNorm_input_shape[2] / unit
        self.ops["RMSNorm"] = {"name":"RMSNorm", "type": "Vector", "ishape":RMSNorm_input_shape, "wshape": RMSNorm_weight_shape, "oshape":RMSNorm_output_shape, "compute":RMSNorm_compute}

        # 2. Q_proj phase
        Proj_input_shape = deepcopy(RMSNorm_output_shape)
        # weight_shape: (hidd_size, hidd_size)
        Proj_weight_shape = [self.config["D_QKV"], self.config["H_QKV"]]
        Proj_output_shape = [Proj_input_shape[0], Proj_input_shape[1], Proj_weight_shape[1]]
        Proj_compute = 2*Proj_input_shape[0]*Proj_input_shape[1]*Proj_weight_shape[0]*Proj_weight_shape[1]/unit
        self.ops["Q_proj"] = {"name":"Q_proj","type": "GEMM", "ishape":Proj_input_shape, "wshape": Proj_weight_shape, "oshape":Proj_output_shape, "compute":Proj_compute}

        # 3. K_proj
        self.ops["K_proj"] = {"name":"K_proj","type": "GEMM", "ishape":Proj_input_shape, "wshape": Proj_weight_shape, "oshape": Proj_output_shape, "compute": Proj_compute}
        # 4. V_proj
        self.ops["V_proj"] = {"name":"V_proj", "type": "GEMM", "ishape": Proj_input_shape, "wshape": Proj_weight_shape, "oshape": Proj_output_shape, "compute": Proj_compute}

        # 5. RoPE(Q) only for each head
        RoPE_input_shape = [self.config["B"], self.config["S"], self.config["H_A"]]
        # split col into each head
        RoPE_input_shape[2] = int(RoPE_input_shape[2]/self.config["N_A"])
        RoPE_weight_shape = [2*RoPE_input_shape[1], RoPE_input_shape[2]]
        RoPE_output_shape = RoPE_input_shape
        RoPE_compute = 3*RoPE_input_shape[0]*RoPE_input_shape[1]*RoPE_input_shape[2]/unit
        self.ops["RoPE(Q)"] = {"name":"RoPE(Q)","type": "Vector", "ishape":RoPE_input_shape, "wshape": RoPE_weight_shape, "oshape": RoPE_output_shape, "compute": RoPE_compute}
        # 6. RoPE(K) only for each head
        self.ops["RoPE(K)"] = {"name":"RoPE(K)", "type": "Vector", "ishape":RoPE_input_shape, "wshape": RoPE_weight_shape, "oshape": RoPE_output_shape, "compute": RoPE_compute}
        # 7. QK^{T}
        QK_input_shape = deepcopy(RoPE_output_shape)
        QK_weight_shape = [QK_input_shape[2], QK_input_shape[1]]
        QK_output_shape = [QK_input_shape[0], QK_input_shape[1], QK_weight_shape[1]]
        QK_compute = 2*QK_input_shape[0]*QK_input_shape[1]*QK_weight_shape[0]*QK_weight_shape[1]/unit
        self.ops["QK^T"] = {"name":"QK^T", "type": "GEMM", "ishape":QK_input_shape, "wshape": QK_weight_shape, "oshape":QK_output_shape, "compute":QK_compute}
        # 8. Softmax
        Softmax_input_shape = deepcopy(QK_output_shape)
        Softmax_weight_shape = None
        Softmax_output_shape = Softmax_input_shape
        Softmax_compute = 5*Softmax_input_shape[0]*Softmax_input_shape[1]*Softmax_input_shape[2]/unit 
        self.ops["Softmax"] = {"name":"Softmax", "type": "Vector", "ishape": Softmax_input_shape, "wshape": Softmax_weight_shape, "oshape": Softmax_output_shape, "compute": Softmax_compute}
        # 9. AV
        AV_input_shape = deepcopy(Softmax_output_shape)
        AV_weight_shape = [Proj_output_shape[1], int(Proj_output_shape[2]/self.config['N_A'])]
        AV_output_shape = [AV_input_shape[0], AV_input_shape[1], AV_weight_shape[1]]
        AV_compute = 2*AV_input_shape[0]*AV_input_shape[1]*AV_weight_shape[0]*AV_weight_shape[1]/unit
        self.ops["AV"] = {"name":"AV", "type": "GEMM", "ishape":AV_input_shape, "wshape": AV_weight_shape, "oshape": AV_output_shape, "compute": AV_compute}
        # 10. Linear 
        Linear_input_shape = [self.config['B'], self.config["S"], self.config["D_O"]]
        Linear_weight_shape = [self.config["D_O"], self.config["H_O"]]
        Linear_output_shape = Linear_input_shape
        Linear_compute = 2*Linear_input_shape[0]*Linear_input_shape[1]*Linear_weight_shape[0]*Linear_weight_shape[1]/unit
        self.ops["Linear"] = {"name":"Linear", "type": "GEMM", "ishape": Linear_input_shape, "wshape": Linear_weight_shape, "oshape": Linear_output_shape, "compute": Linear_compute}
        # 11. ResAdd
        ResAdd_input_shape = deepcopy(Linear_output_shape)
        ResAdd_weight_shape = deepcopy(Linear_weight_shape)
        ResAdd_output_shape = ResAdd_input_shape
        ResAdd_compute = ResAdd_input_shape[0]*ResAdd_input_shape[1]*ResAdd_input_shape[2]/unit
        self.ops["ResAdd"] = {"name":"ResAdd", "type": "Vector", "ishape": ResAdd_input_shape, "wshape": ResAdd_weight_shape, "oshape": ResAdd_output_shape, "compute": ResAdd_compute}
        # 12. RMSNorm
        self.ops["RMSNorm2"] = {"name":"RMSNorm2", "type": "Vector", "ishape":RMSNorm_input_shape, "wshape": RMSNorm_weight_shape, "oshape":RMSNorm_output_shape, "compute":RMSNorm_compute}
        # 13. FFNup
        FFNup_input_shape = [self.config["B"], self.config["S"], self.config["D_FU"]]
        FFNup_weight_shape = [self.config['D_FU'], self.config["H_FU"]]

        FFNup_output_shape = [FFNup_input_shape[0], FFNup_input_shape[1], FFNup_weight_shape[1]]
        FFNup_compute = 2*FFNup_input_shape[0]*FFNup_input_shape[1]*FFNup_weight_shape[0]*FFNup_weight_shape[1]/unit
        self.ops["FFNup"] = {"name":"FFNup", "type": "GEMM", "ishape":FFNup_input_shape, "wshape": FFNup_weight_shape, "oshape": FFNup_output_shape, "compute": FFNup_compute}
        # 14. FFNgate
        self.ops["FFNgate"] = {"name":"FFNgate", "type": "GEMM", "ishape":FFNup_input_shape, "wshape": FFNup_weight_shape, "oshape": FFNup_output_shape, "compute": FFNup_compute}
        # 15. SiLU
        SiLU_input_shape = deepcopy(FFNup_output_shape)
        SiLU_weight_shape = None
        SiLU_output_shape = SiLU_input_shape
        SiLU_compute = 4*SiLU_input_shape[0]*SiLU_input_shape[1]*SiLU_input_shape[2]/unit 
        self.ops["SiLU"] = {"name":"SiLU", "type": "Vector", "ishape": SiLU_input_shape, "wshape": SiLU_weight_shape, "oshape": SiLU_output_shape, "compute": SiLU_compute}
        # 16. Hadamard
        Hadamard_input_shape = deepcopy(SiLU_output_shape)
        Hadamard_weight_shape = [FFNup_output_shape[1], FFNup_output_shape[2]]
        Hadamard_output_shape = Hadamard_input_shape
        Hadamard_compute = Hadamard_input_shape[0]*Hadamard_input_shape[1]*Hadamard_input_shape[2]/unit 
        self.ops["Hadamard"] = {"name":"Hadamard", "type": "Vector", "ishape": Hadamard_input_shape, "wshape": Hadamard_weight_shape, "oshape": Hadamard_output_shape, "compute": Hadamard_compute}
        # 17. FFNdown
        FFNdown_input_shape = deepcopy(Hadamard_output_shape)
        FFNdown_weight_shape = [self.config["D_FD"], self.config["H_FD"]]
        FFNdown_output_shape = [FFNdown_input_shape[0], FFNdown_input_shape[1], FFNdown_weight_shape[1]]
        FFNdown_compute = 2*FFNdown_input_shape[0]*FFNdown_input_shape[1]*FFNdown_weight_shape[0]*FFNdown_weight_shape[1]/unit
        self.ops["FFNdown"] = {"name":"FFNdown", "type": "GEMM", "ishape":FFNdown_input_shape, "wshape": FFNdown_weight_shape, "oshape": FFNdown_output_shape, "compute": FFNdown_compute}
        # 18. ResAdd
        self.ops["ResAdd2"] = {"name":"ResAdd2", "type": "Vector", "ishape": ResAdd_input_shape, "wshape": ResAdd_weight_shape, "oshape": ResAdd_output_shape, "compute": ResAdd_compute}


if __name__ == "__main__":
    input_path = "./input/transformer/input0.json"
    output_path = "./output/transformer/llama.xlsx"
    config = load_config(input_path)
    llama7b = Llama_block(config)
    llama7b.print_ops()
    llama7b.save_ops(output_path)







        












