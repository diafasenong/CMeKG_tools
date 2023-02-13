'''
完成数据集的准备
'''
import os
import re
from torch.utils.data import Dataset,DataLoader
from ws_lib import ws,max_len,batch_size
import torch

def tokenization(text):
    text=re.sub('<.*?>',' ',text)
    filters=["\?","\.","\t","\n","\x97","\x96","\""]
    text=re.sub('|'.join(filters),' ',text)
    tokens=[item.strip().lower() for item in text.split(' ') if item != '']
    return tokens


class ImdbDataset(Dataset):
    def __init__(self,isTrain=True) -> None:
        base_path = '/Users/wangjialin/projects/CMeKG_tools/torch'
        # 获取文件路径
        self.train_data_path=os.path.join(base_path,'data/aclImdb/train')
        self.test_data_path=os.path.join(base_path,'data/aclImdb/test')
        data_path = self.train_data_path if isTrain else self.test_data_path
        # 将neg和pos的文件名统一放入列表
        temp_data_path=[os.path.join(data_path,'pos'),os.path.join(data_path,'neg')]
        self.total_file_path=[]
        for path in temp_data_path:
            file_name_list=os.listdir(path)
            file_path_list=[os.path.join(path,item) for item in file_name_list]
            self.total_file_path.extend(file_path_list)
        
    def __getitem__(self, index):
        file_name= self.total_file_path[index]
        # print(file_name)
        # 获取label
        label_str = str(file_name).split('/')[-2]
        # print(label_str)
        label = 0 if label_str == 'neg' else 1
        # 获取内容
        real_content = open(file_name).read()
        # print(real_content)
        
        return tokenization(real_content),label
    
    def __len__(self):
        return len(self.total_file_path)
    
    
def collate_fn(batchs):
    # ret=list(zip(*batchs))
    # return ret
    content,label = list(zip(*batchs))
    content = [ws.transform(i,max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content,label
    
def get_dataloader(isTrain=True):
    imdb_dataset = ImdbDataset(isTrain)
    imdb_dataloader = DataLoader(imdb_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return imdb_dataloader
    

if __name__=="__main__":
    imdb_dataloader = get_dataloader()
    for idx,(content,label) in enumerate(imdb_dataloader):
        print(idx)
        print(content)
        print(label)
        break