'''
将保存的类文件加载到可用的类中
'''
import pickle

ws = pickle.load(open('/Users/wangjialin/projects/CMeKG_tools/mymodel/ws.pkl','rb'))

# 数据最大长度
max_len = 20
# 批量宽度
batch_size = 128