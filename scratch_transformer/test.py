from helpers import get_param_num
import torch 
import torch.nn as nn
import torch.nn.functional as F
from encoderBlock import EncoderBlock
from decoderBlock import DecoderBlock
from feedForward import Feed_Forward
from Gpt import GPT
from attention_block import Attention_Block
import numpy as np 
from transformer import Transformer
from decoder_only_Block import Gpt_Block



def test1():
    
    enc = EncoderBlock(10000,20)
    total_params = sum(p.numel() for p in enc.parameters())
    print(f"Total number of parameters: {total_params}")
    test = torch.rand((20,10000))
    enc(test)
    




def test0():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # Use GPU
    else:
        device = torch.device("cpu")   # Use CPU
    return device

def test2():
    device = test0()
    
    att = Attention_Block()
    
    test = torch.rand((32,7,512))
    
    
    b = att(test)
    # get_param_num(att)


    ff = Feed_Forward(512)
    
    # print (b)

    b = ff(b)
    

    
    

    
    # print(b)
    













def test3():
    dec = DecoderBlock()
    total_params = sum(p.numel() for p in dec.parameters())
    print(f"Total number of parameters: {total_params}")
    decIn = torch.rand((5,512))
    encOut = torch.rand((5,512))
    a = dec(decIn,encOut)
    print(a.shape)
    
def test6():
    dec = EncoderBlock(16)
    total_params = sum(p.numel() for p in dec.parameters())
    print(f"Total number of parameters: {total_params}")
    encIn = torch.rand((5,16))
    a = dec(encIn)

    print(a.shape)


def test4():
    # Vocab_size,EMBEDDING_DIM,10,32,32,32
    # device = test0()
    device = "cuda"
    # print(device)
    T = Transformer(10000).to(device)
    total_params = sum(p.numel() for p in T.parameters())
    # print(f"Total number of parameters: {total_params}")
    input_ = torch.randint(10,100,size=(32,10)).to(device)

    
    
    dec_input = torch.randint(0,100,size=(32,2)).to(device)
    # print(dec_input.shape)
    
    a = T(input_,dec_input)
    




def test5():
    # Vocab_size,6,EMBEDDING_DIM,10,32,32,32
    gpt = GPT(vocab_size=20000,).to("cuda")



    total_params = sum(p.numel() for p in gpt.parameters())
    print(f"Total number of parameters: {total_params}")

    input_ = torch.randint(10,100,size=(128,299)).to("cuda")
    print(input_)
    # decIn = decIn.type(torch.float)
    # print(decIn)
    
    a = gpt(input_)
    
    
    



def test11():
    # Vocab_size,6,EMBEDDING_DIM,10,32,32,32
    gpt = Gpt_Block().to("cuda")



    total_params = sum(p.numel() for p in gpt.parameters())
    print(f"Total number of parameters: {total_params}")

    input_ = torch.rand(size=(32,432,512)).to("cuda")
    # print(input_)
    # decIn = decIn.type(torch.float)
    # print(decIn)
    
    a = gpt(input_)
    
    
    
    
    


if __name__ =="__main__":
    import timeit
    


    n = 5

    # calculate total execution time
    result = timeit.timeit(stmt='test5()', globals=globals(), number=n)


    # calculate the execution time
    # get the average execution time
    print(f"Execution time is {result / n} seconds")
    