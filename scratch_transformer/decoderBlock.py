import torch.nn as nn
import torch.nn.functional as F
from attention_block import Attention_Block
from feedForward import  Feed_Forward


# for the masked attention we just need to make the  sembedding of the mask tokens to be neg inf.
class DecoderBlock(nn.Module):

    def __init__(self,input_size=512,heads=8):
        super(DecoderBlock,self).__init__()

        self.d_model = input_size
        self.heads = heads
        self.head_dim = self.d_model//heads




        self.masked_att = Attention_Block(input_size,heads,mask=1)
        self.maskedQ = nn.Linear(input_size,self.heads*self.head_dim,bias=False)
        self.encoderV = nn.Linear(input_size,self.heads*self.head_dim,bias=False)
        self.encoderK = nn.Linear(input_size,self.heads*self.head_dim,bias=False)

        self.att = Attention_Block(input_size,heads,enc_dec=1)
        

        self.ff = Feed_Forward(input_size)

        self.norm = nn.LayerNorm(input_size)
        




        
      

        

    
    def forward(self,x,enc_out):
        

        # positional encoding

        
        #masked
        output  = self.masked_att(x)
        output += x
        output_masked  = self.norm(output)
        
    
        #cross attention
        masked_Q = self.maskedQ(output_masked)
        encV = self.encoderV(enc_out)
        encK = self.encoderK(enc_out)

        output = self.att(x.shape[0],Q=masked_Q,V=encV,K=encK)
    
        output+= output_masked
        output_att = self.norm(output)
        
        
        #FF
        output = self.ff(output_att)
        output+= output_att 
        output = self.norm(output)



    
    



        return output

        

