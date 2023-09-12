import torch.nn as nn
import torch.nn.functional as F



class Feed_Forward(nn.Module):
    
    def __init__(self,input_size):
        super(Feed_Forward,self).__init__()
        self.l1 = nn.Linear(input_size,4*input_size)
        self.l2 = nn.Linear(4*input_size,input_size)

    def forward(self,x):
        
        output = self.l1(x)
        output  = F.relu(output)
        output = self.l2(output)
        return output


