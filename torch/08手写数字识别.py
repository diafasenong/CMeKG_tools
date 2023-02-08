from torchvision.datasets import MNIST
from torchvision.transforms import Normalize,ToTensor,Compose

from torch.utils.data import DataLoader
# 模型基类
import torch.nn as nn
import torch.nn.functional as F
# 优化器类
from torch.optim import Adam
import torch
import os

# 其他类
import numpy as np

# 配置参数
BATCH_SIZE = 128

# 0准备数据
def get_dataloader(train=True):
    # 数据转换方法
    transform_fn = Compose(
        [
            ToTensor(),
            Normalize(mean=(0.1307,),std=(0.3081,))
            # mean和std的形状和通道数相同
        ]
    )

    dataset_data = MNIST(root='/Users/wangjialin/projects/CMeKG_tools/torch/data/',train=train,transform=transform_fn)
    data_loader = DataLoader(dataset=dataset_data,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader

# 1构建模型
class My_Model(nn.Module):
    
    def __init__(self) -> None:
        super(My_Model,self).__init__()
        self.fn1 = nn.Linear(1*28*28,28)
        self.fn2 = nn.Linear(28,10)
        
    def forward(self,input):
        '''
        input: [batch_size,1,28,28]
        '''
        x = input.view([-1,1*28*28])
        # -1处自动获取行数，方便后续评估更改batch_size
        # x = input.view([input.size(0),1*28*28])
        x = self.fn1(x)
        x = F.relu(x)
        out = self.fn2(x)
        # return out
        # 为了计算损失
        return F.log_softmax(out,dim=-1)

model = My_Model()
optimizer = Adam(model.parameters(),lr=0.001)
if os.path.exists('./mymodel/model.pkl'):
    model.load_state_dict(torch.load('./mymodel/model.pkl'))
    optimizer.load_state_dict(torch.load('./myresult/optimizer.pkl'))

def train(epoch):
    '''实现训练的过程'''
    mode = True
    data_loader = get_dataloader(train=mode)
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()   #梯度置零
        output = model(input)   #调用模型，获得预测值
        loss = F.nll_loss(output,target)    #得到损失
        loss.backward()         #反向传播
        optimizer.step()        #更新参数
        if idx%20 ==0:
            print(epoch,idx,loss.item())
            
        # 模型的保存
        if idx%100 ==0:
            torch.save(model.state_dict(),'./mymodel/model.pkl')
            torch.save(optimizer.state_dict(),'./myresult/optimizer.pkl')
            
def test():
    loss_list=[]
    acc_list=[]
    test_dataloader = get_dataloader(train=False)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            outp = torch.tensor(output)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            # 计算准确率
            pred=outp.max(dim=-1)[-1]
            pre=torch.tensor(pred)
            cur_acc=pre.eq(target).float().mean()
            acc_list.append(cur_acc)
    loss_avg=np.mean(loss_list)
    acc_avg=np.mean(acc_list)
    print('平均准确率,平均损失:',acc_avg,loss_avg)
            
if __name__ == '__main__':
    # for i in range(3):  #训练轮数
    #     train(i)
    
    test()