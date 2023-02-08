import numpy as np
import torch

from torchvision import transforms

data=np.random.randint(low=0,high=255,size=12)
print(data)
img=data.reshape([2,2,3])
print(img)
# 图片转三维张量
dest=transforms.ToTensor()(img)
print(dest)
print('*'*10)
ts=torch.tensor(img)
# 图片转三维张量等同于对张量permute转换
des=ts.permute(2,0,1)
print(des)