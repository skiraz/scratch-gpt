
import torch.nn as nn
import torch.nn.functional as F
from attention_block import Attention_Block
from feedForward import  Feed_Forward




class EncoderBlock(nn.Module):
    def __init__(self,input_size=512,heads=8):
        super(EncoderBlock,self).__init__()
        self.att = Attention_Block(input_size,heads,mask=None)

        self.ff  = Feed_Forward(input_size)

        self.norm = nn.LayerNorm(input_size)


        

    def forward(self,x):
        

        # positional encoding


        
        
        output  = self.att(x)
        
        
        output  += x
        output = self.norm(output)
        

        output = self.ff(x)
        output += x 
        output = self.norm(output)

        return output

        

