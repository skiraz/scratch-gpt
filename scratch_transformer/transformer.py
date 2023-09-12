import torch
import torch.nn as nn 
from helpers import *
import torch.nn.functional as F

from encoderBlock import EncoderBlock
from decoderBlock import DecoderBlock

# for the masked attention we just need to make the  sembedding of the mask tokens to be neg inf.
class Transformer(nn.Module):

    def __init__(self,vocab_size,enc_stack=6,dec_stack=6,Q=None,K=None,V=None,d_model=512,heads=8):
        super(Transformer,self).__init__()
    

        
        
        self.embedding_layer = nn.Embedding(vocab_size, d_model)

        self.encoder = EncoderBlock(d_model,heads)
        self.decoder = DecoderBlock(d_model,heads)
        
        self.probs = nn.Linear(d_model,vocab_size)


    def forward(self,x,dec_input):
        
        
        x = self.embedding_layer(x)
        
        

        output = self.encoder(x)
        
        #  auto regressive ->

        dec_input = self.embedding_layer(dec_input)
        

        output = self.decoder(dec_input,output)
        
        
        
        

        output = self.probs(output)
        output = F.softmax(output,dim=1)
 
        
        

        return output

        





