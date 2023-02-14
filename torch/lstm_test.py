'''
lstm使用示例
'''

import torch.nn as nn
import torch

# 初始化参数
batch_size = 3
sequence_len = 4
embedding_size = 100
embedding_dim = 5
hidden_size = 6
num_layers = 10
batch_first = True
bidirectional = False

# 创建输入数据
input = torch.randint(low=0,high=10,size=[batch_size,sequence_len])
print('创建输入数据')
print(input.size())
print('torch.Size([%d, %d])'%(batch_size,sequence_len))
# print(input)

# 使用embedding
embedding = nn.Embedding(embedding_size,embedding_dim)
input_embeded = embedding(input)
print('embedding处理后的数据')
print(input_embeded.size())
print('torch.Size([%d, %d, %d])'%(batch_size,sequence_len,embedding_dim))
# print(input_embeded)

# lsmt处理
lsmt = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=batch_first,bidirectional=bidirectional)
output,(h_n,c_n) = lsmt(input_embeded)
print('lsmt输出')
d = 2 if bidirectional else 1
print(output.size())
print('torch.Size([%d, %d, %d])'%(batch_size,sequence_len,hidden_size*d))
# print(output)
print('h_n数据')
print(h_n.size())
print('torch.Size([%d, %d, %d])'%(1*num_layers*d,batch_size,hidden_size))
# print(h_n)
print('c_n数据')
print(c_n.size())
print('torch.Size([%d, %d, %d])'%(1*num_layers*d,batch_size,hidden_size))
# print(c_n)
print('*'*50)
a = output[:,-1,:]
print(a)
print('-'*50)
b = h_n[-1,:,:]
print(b)
print(a==b)
