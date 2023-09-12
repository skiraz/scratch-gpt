
import torch.nn as nn
import torch.nn.functional as F
from decoder_only_Block import Gpt_Block











class GPT(nn.Module):
    def __init__(self,vocab_size,d_model=512,heads=8,infernece_mode=0,N=6):
        super(GPT,self).__init__()

        self.heads = heads
        self.d_model = d_model
        self.inference_mode = infernece_mode
        self.N = N
        self.embedding_layer = nn.Embedding(vocab_size, d_model)
        self.stack = [ Gpt_Block(d_model=self.d_model,heads=self.heads) for n in range(self.N) ]
        self.stack = nn.Sequential(*self.stack)
        self.norm =  nn.LayerNorm(self.d_model)

        

        
        
        self.probs = nn.Linear(d_model,vocab_size)
        self.probs.weight = self.embedding_layer.weight



    def forward(self,x):
        

        # positional encoding
    

        
        output = self.embedding_layer(x)
   

        for block in self.stack:

            output += block(output) 
            output = self.norm(output)

        
        
        output = self.probs(output)

        if self.inference_mode:
            ## make it so that it outputs the last word only and make it auto regressive
            pass
        output = F.softmax(output,dim=1)

        return output
        

