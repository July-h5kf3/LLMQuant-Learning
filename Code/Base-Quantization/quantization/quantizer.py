import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Quantizer(nn.Module):
    def __init__(self,bit,observer,ptq,sign=False):
        super(Quantizer,self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
    
    def update_qparams(self,tensor):
        raise NotImplementedError
    
    def forward(self,tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)
        quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        #这里进行Detach是因为round函数无法求导
        fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale
        return fake_quant_tensor

class AsymmetricQuanizer(Quantizer):
    """
    非对称量化
    """
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer,self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq

        if self.observer.level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
        elif self.observer.level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),
            )
        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
                            )
    def update_qparams(self, inputs):
        scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                        (self.quant_min - self.observer.min_val / scale)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        
class AdaRoundQuantizer(Quantizer):
    def __init__(self,bit,observer,ptq,sign=False,round_mode = 'learned_hard_sigmoid'):
        super(Quantizer,self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        self.round_mode = round_mode
        self.alpha = None
        self.ada_init = None
        self.soft_targets = True
        self.gamma,self.zeta = -0.1,1.1
        self.beta = 2 / 3

        if self.observer.level == "L":
            self.register_buffer("scale",torch.ones(1),dtype = torch.float32)
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),
                            )
        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
                            )
    
    def update_qparams(self, inputs):
        scale = (self.observer.max_val -self.observer.min_val) / (self.quant_max - self.quant_min)
        zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                        (self.quant_min - self.observer.min_val / scale)
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
    
    def init_alpha(self,x):
        scale = self.scale

        x_floor = torch.floor(x / scale)
        if self.round_mode == "learned_hard_sigmoid":
            print("Init alpha to be FP32")
            rest = (x / scale) - x_floor
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)#Sigmoid(alpha) = rest
            """
            这里对应原论文的W = s*clip[W/s+h(V)]其中V就是这里的alpha是一个可学习的参数
            我们希望h(alpha) = rest,即h_Sigmoid(alpha) = rest
            那么有Sigmoid(alpha) * (zeta - gamma) + gamma = rest
            (rest - gamma) / (zeta - gamma) = Sigmoid(alpha) = 1 / (1 + exp(-alpha))
            exp(-alpha) = ((zeta -gamma) - rest + gamma) / (rest - gamma)
            alpha = -log((zeta - rest) / (rest - gamma))
            """
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    def quant(self,inputs,scale = None,zero_point = None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        
        if self.round_mode == "nearest":
            x_int = torch.round(inputs / scale)
        elif self.round_mode == "learned_hard_sigmoid":
            x_floor = torch.floor(inputs / scale)
            if self.get_soft_targets:
                x_int = x_floor + self.get_soft_target()
            else:
                print('test test test')
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')
        outputs = x_int + zero_point
        outputs = outputs.round().clamp(self.quant_min,
                                        self.quant_max)
        return outputs

    def dequantize(self,inputs,scale = None,zero_point = None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point

        outputs = (inputs - zero_point) * scale
        return outputs 

    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)
        
        if not self.ada_init:
            self.init_alpha(tensor.clone())
            self.ada_init = True
        
        quant_tensor = self.quant(tensor)
        fake_quant_tensor = self.dequantize(quant_tensor)

        return fake_quant_tensor
    
