from ws_word_sequence import Word2Sequence
from ws_dataset import tokenization

import pickle   # 保存语料包
from tqdm import tqdm   # 显示迭代进度

import os

if __name__=='__main__':
    ws = Word2Sequence()
    base_path = '/Users/wangjialin/projects/CMeKG_tools/torch'
    # 获取文件路径
    train_data_path=os.path.join(base_path,'data/aclImdb/train')
    all_train_data_path=[os.path.join(train_data_path,'pos'),os.path.join(train_data_path,'neg')]
    for data_path in all_train_data_path:
        file_names = os.listdir(data_path)
        # print(data_path)
        file_paths = [os.path.join(data_path,file_name) for file_name in file_names if file_name.endswith('txt')]
        for file_path in tqdm(file_paths):
            file_content = open(file_path).read()
            sentences = tokenization(file_content)
            ws.fit(sentences)
            
    ws.build_vocab(min=10,max_features=10000)
    # 保存类数据
    pickle.dump(ws,open('/Users/wangjialin/projects/CMeKG_tools/mymodel/ws.pkl','wb'))
    print(len(ws))
    