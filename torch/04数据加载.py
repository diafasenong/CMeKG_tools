import torch
from torch.utils.data import Dataset,DataLoader

data_path = r'/Users/wangjialin/projects/CMeKG_tools/torch/data/SMSSpamCollection'

class My_Data(Dataset):
    
    def __init__(self) -> None:
        self.lines = open(data_path).readlines()
    
    def __getitem__(self, index):
        item= self.lines[index].strip()
        ar=str(item).split('\t')
        return ar[0],ar[1]
    
    def __len__(self):
        return len(self.lines)
      

if __name__ == '__main__':
    mydata = My_Data()
    print('总数据数:',len(mydata))
    print('第一条数据:',mydata[0])
    for index,item in enumerate(mydata):
        if index > 5:
            break
        print(str(index+1)+str(item))
    my_loader=DataLoader(dataset=mydata,batch_size=2,shuffle=True)
    for index,item in enumerate(my_loader):
        print(item)
        break
    
    