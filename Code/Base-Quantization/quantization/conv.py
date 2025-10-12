import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.observer import MinMaxObserver,EMAMinMaxObserver
from quantization.quantizer import AsymmetricQuanizer,AdaRoundQuantizer

class QConv2d(nn.Conv2d):
    def __init__(
            self,
            ptq,
            level,
            adaround,
            in_channels,out_channels,kernel_size,stride = 1,padding = 0,dilation = 1,
            groups = 1,bias = True,padding_mode = 'zero',
            bit = 8,
            sign = False,
            **kwargs
    ):
        super(QConv2d,self).__init__(
            in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode
        )
        self.ptq = ptq

        if adaround:
            self.weight_quantizer = AdaRoundQuantizer(bit=bit,observer=MinMaxObserver(out_channels,level),ptq=ptq,sign=sign)
            self.input_quantizer = AsymmetricQuanizer(bit=bit,observer=EMAMinMaxObserver(out_channels,"L"),ptq=ptq,sign=sign)
        else:
            self.weight_quantizer = AsymmetricQuanizer(bit=bit,observer=MinMaxObserver(out_channels,level),ptq=ptq,sign=sign)
            self.input_quantizer = AsymmetricQuanizer(bit=bit,observer=EMAMinMaxObserver(out_channels,"L"),ptq=ptq,sign=sign)
    
    def forward(self,input):
        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.conv2d(
            input,weight_quant,self.bias,self.stride,self.padding,self.dilation,self.groups
        )
        return output