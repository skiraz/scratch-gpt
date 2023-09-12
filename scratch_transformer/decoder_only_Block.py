import torch.nn as nn
import torch.nn.functional as F
from attention_block import Attention_Block
from feedForward import  Feed_Forward


# for the masked attention we just need to make the  sembedding of the mask tokens to be neg inf.
class Gpt_Block(nn.Module):

    def __init__(self,d_model=512,heads=8):
        super(Gpt_Block,self).__init__()

        self.d_model = d_model
        self.heads = heads
        self.head_dim = self.d_model//heads




        self.masked_att = Attention_Block(self.d_model,heads,mask=1)
        

        # self.att = Attention_Block(self.d_model,heads,enc_dec=0)
        

        self.ff = Feed_Forward(self.d_model)

        self.norm = nn.LayerNorm(self.d_model)
        




        
      

        

    
    def forward(self,x):
        

        # positional encoding

        
        #masked
        output  = self.masked_att(x)
        output += x
        output_masked  = self.norm(output)
        


        # for some reason u dont need the cross attention  
        # # normal
        # output  = self.att(output_masked)
        # output  += output_masked
        # output_att = self.norm(output)
        
        
        #FF
        output = self.ff(output_masked)
        output+= output_masked 
        output = self.norm(output)



    
    



        return output

        

