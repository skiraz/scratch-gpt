import sys
sys.path.append(r"D:\projects\TORCH\transformer")
from transformer import Transformer


with open(r"D:\projects\TORCH\shakespear LM\data\alllines.txt","w") as f :
    data = f.readlines()

print(data)