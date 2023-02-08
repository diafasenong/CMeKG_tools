'''
完成数据集的准备
'''
import os

from torch.utils.data import Dataset,DataLoader

class ImdbDataset(Dataset):
    def __init__(self,isTrain=True) -> None:
        # 获取文件路径
        self.train_data_path='./data/aclImdb/train'
        self.test_data_path='./data/aclImdb/test'
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
        return file_name
    
    def __len__(self):
        return len(self.total_file_path)
    
if __name__=="__main__":
    imdb = ImdbDataset()
    print(imdb[0])