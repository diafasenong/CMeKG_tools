'''
构建词典，将句子转为数字列表，将数字列表转为句子
'''

class Word2Secquence(object):
    # 设置需要的默认字符
    UNK_TAG = 'UNK' #未识别的默认字符
    PAD_TAG = 'PAG' #填充的字符
    
    UNK = 0
    PAD = 1
    
    def __init__(self) -> None:
        # 创建词料字典表
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        # 设置统计字典
        self.count = {}
        
    # 把句子填充到统计字典中
    '''
    :param sentence:[word1,word2,word3...]
    '''
    def fit(self,sentence):
        for word in sentence:
            print(word)
            self.count[word] = self.count.get(word,0)+1
         
    # 将统计字典的数据添加到词料字典表
    '''
    :param min:int 最小词频
    :param max:int 最大词频
    :param max_features:int 最大词料数量
    '''
    def build_vocab(self,min=5,max=None,max_features=None):
        # 过滤统计字典数据
        if min is not None:
            self.count = {k:v for k,v in self.count.items() if v>min}
        if max is not None:
            self.count = {k:v for k,v in self.count.items() if v<max}
        if max_features is not None:
            temp = sorted(self.count.items(),key=lambda i:i[-1],reverse=True)[:max_features]
            self.count = dict(temp)
            
        # 把所有的统计字典放进词料字典
        for word in self.count:
            self.dict[word] = len(self.dict)
            
        # 获取一个反转的dict
        self.inverse_dict = dict(zip(self.dict.values(),self.dict.keys()))
        
    # 将句子转为序列
    '''
    :param sentence:[word1,word2,word3...]
    :param max_len:int
    :return [1,3,5,2,4...]
    '''
    def transform(self,sentence,max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                # 补充
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))
                
            if max_len < len(sentence):
                # 剪裁
                sentence = sentence[:max_len]
        # 转位数字序列
        # return [self.dict[word] for word in sentence]
        return [self.dict.get(word,self.UNK) for word in sentence]
    
    # 将序列转为句子
    '''
    :param indices:[1,3,5,2,4...]
    :return [word1,word2,word3...]
    '''
    def inverse_transform(self,indices):
        # return [self.inverse_dict[num] for num in indices]
        return [self.inverse_dict.get(num) for num in indices]

if __name__=='__main__':
    ws = Word2Secquence()
    ws.fit(['今天','是','星期五'])
    ws.fit(['今天','是','晴天'])
    ws.build_vocab(min=0)
    print(ws.dict)
    print('*'*10)
    print(ws.inverse_dict)
    sentence=['今天','是','个','大','晴天']
    ret = ws.transform(sentence,max_len=10)
    print(ret)
    ret = ws.inverse_transform(ret)
    print(ret)
    