'''
定义模型
'''


from ws_lib import ws,max_len
# 导入数据集
from ws_dataset import get_dataloader
# 模型基类
import torch
import torch.nn as nn
# 激活函数，损失函数等
import torch.nn.functional as F
# 优化器类
from torch.optim import Adam
# 帮助类
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self) -> None:
        super(MyModel,self).__init__()
        # 将序列embedding处理
        self.embedding = nn.Embedding(len(ws),100)
        self.fc = nn.Linear(max_len*100,2)
        
    def forward(self,input):
        '''
        :param input:[batch_size,max_len,100]
        '''
        x = self.embedding(input)
        x = torch.tensor(x).view([-1,max_len*100])
        out = self.fc(x)
        return F.log_softmax(out,dim=-1)
    
model = MyModel()
optim = Adam(model.parameters(),lr=1e-3)
def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader()):
        # 梯度置零
        optim.zero_grad()
        output = model(input)
        # 计算损失
        loss = F.nll_loss(output,target)
        loss.backward()
        optim.step()
        # print(loss.item())
        
if __name__ == '__main__':
    for i in range(1):
        train(i)