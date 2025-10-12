import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization.quantizer import AdaRoundQuantizer,AsymmetricQuanizer
from quantization.observer import MinMaxObserver,EMAMinMaxObserver

class QLinear(nn.Linear):
    def __init__(self,ptq,level,adaround,in_features,out_features,bias,bit,sign=True,**kwargs):
        super(QLinear,self).__init__(in_features,out_features,bias)
        self.ptq = ptq
        if level == "L":
            self.fc_level = "L"
        elif level == "C":
            self.fc_level = "FC"
        
        if adaround:
            self.weight_quantizer = AdaRoundQuantizer(bit=bit,observer=MinMaxObserver(out_features,level),ptq=ptq,sign=sign)
            self.input_quantizer = AsymmetricQuanizer(bit=bit,observer=EMAMinMaxObserver(out_features,"L"),ptq=ptq,sign=sign)
        else:
            self.weight_quantizer = AsymmetricQuanizer(bit=bit,observer=MinMaxObserver(out_features,level),ptq=ptq,sign=sign)
            self.input_quantizer = AsymmetricQuanizer(bit=bit,observer=EMAMinMaxObserver(out_features,"L"),ptq=ptq,sign=sign)
    
    def forward(self, input):
        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.linear(input,weight_quant,self.bias)
        return output