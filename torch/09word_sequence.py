'''
实现的是：构建词典，将句子转为数字列表，将数字列表转为句子
'''

class Word2Sequence(object):
    UNK_TAG = 'UNK' #默认字符
    PAD_TAG = 'PAD' #填充字符
    
    UNK = 0
    PAD = 1
    
    
    def __init__(self) -> None:
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        
        self.count = {} #统计词频
        
    def fit(self,sentence):
        '''
        把单个句子保存到dict中
        :param sentence: [word1,word2,word3...]
        '''
        for word in sentence:
            print(word)
            self.count[word] = self.count.get(word,0) + 1
            
    def build_vocab(self,min=0,max=None,max_features=None):
        '''
        生成词典
        :param min:最小出现的次数
        :param max:最大出现的次数
        :param max_features:一共保留多少个词语
        '''
        # 删除count中词频小于min的word
        if min is not None:
            self.count = {word:value for word,value in self.count.items() if value > min}
        # 删除count中词频大于max的word
        if max is not None:
            self.count = {word:value for word,value in self.count.items() if value < max}
        # 限制保留的词语数
        if max_features is not None:
            temp = sorted(self.count.items(),key=lambda x:x[-1],reverse=True)[:max_features]
            self.count = dict(temp)
            
        # 把所有词编号放进字典中
        for word in self.count:
            self.dict[word] = len(self.dict)
            
        # 得到一个反转的dict字典
        self.inverse_dict=dict(zip(self.dict.values(),self.dict.keys()))
        
    def transform(self,sentence,max_len=None):
        '''
        把句子转化为序列
        :param sentence:[word1,word2...]
        :param max_len: int,对句子进行填充获取剪裁
        '''
        # for word in sentence:
        #     self.dict.get(word,self.UNK)
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))    #填充
            if max_len < len(sentence):
                sentence = sentence[:max_len]   #裁剪
        # 转为数字序列
        return [self.dict.get(word,self.UNK) for word in sentence]
    
    def inverse_transform(self,indices):
        '''
        把序列转化为句子
        :param indices:[1,2,3,4...]
        '''
        return [self.inverse_dict.get(idx) for idx in indices]
    

if __name__ =='__main__':
    ws = Word2Sequence()
    ws.fit(['我','是','谁','呀'])
    ws.fit(['我','是','你'])
    ws.build_vocab()
    print(ws.dict)