import torch
import torch.nn as nn
import torch.nn.functional as F



# for the masked attention we just need to make the  embedding of the mask tokens to be neg inf.
class Attention_Block(nn.Module):

    def __init__(self,d_model=512,heads=8,mask=None,enc_dec=0):
        super(Attention_Block,self).__init__()
        self.enc_dec = enc_dec
        self.mask = mask
        self.heads = heads
        self.d_model = d_model
        self.head_dimension = self.d_model//self.heads
        

        # self.head_dim = self.d_model//self.heads
        assert((self.d_model%self.heads)==0),"model dims should be div by heads num"


        self.Q = nn.Linear(d_model,self.head_dimension*self.heads,bias=False)
        self.V = nn.Linear(d_model,self.head_dimension*self.heads,bias=False)
        self.K = nn.Linear(d_model,self.head_dimension*self.heads,bias=False)
        
        #   [512,512//heads for each ] 



        self.conc_heads = nn.Linear(self.d_model,d_model)



    def forward(self,embedding=None,Q=None,K=None,V=None):
        
        # this code is to extract the time sequence // ithink its useless from now 
        if isinstance(embedding,int): 
            T = embedding             

        elif embedding is not None:
            T = embedding.shape[0]

        
        if not self.enc_dec:
                Q = self.Q(embedding)
                V = self.V(embedding)
                K = self.K(embedding)
        
        batch = Q.shape[0]
         
        # query_len,value_len,key_len = Q.shape[1],V.shape[1],K.shape[1]
        query_T,value_T,key_T = Q.shape[1],V.shape[1],K.shape[1]
        query_dim,value_dim,key_dim = Q.shape[-1]//self.heads,V.shape[-1]//self.heads,K.shape[-1]//self.heads
        
        
        #compute attention scores 
        Q = Q.reshape((batch,self.heads,query_T,query_dim))
        K = K.reshape((batch,self.heads,value_T,key_dim))
        V = V.reshape((batch,self.heads,key_T,value_dim))
        
        
        self.att_scores = Q @ K.mT
        
        #scale 
        self.att_scores_scaled = self.att_scores/ (key_dim**(1/2))
        

        #mask
        if self.mask:
            trill = torch.tril(torch.ones((batch,self.heads ,query_T,query_T ))).to("cuda")
            self.att_scores_scaled = self.att_scores_scaled.masked_fill(trill==0,float("-inf")).to("cuda")
            

            
    
        
        
        #softmax  
        self.att_scores_scmx = F.softmax(self.att_scores_scaled,dim=2)
        
        
        
        #mat mul
        
             
        
        output = self.att_scores_scmx @ V
        
        

        output = output.reshape((batch,query_T,self.heads*self.head_dimension))
        

        #heads concat
        output = self.conc_heads(output)
        # if self.mask:
        #      print(output)
            
        

        return output

    