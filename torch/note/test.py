from time import time

def run_time(func):
    def wrap(a,b):
        t1 = time()
        func(a,b)
        t2 = time()
        print(t2-t1)
        return 'hehe'
    return wrap

@run_time
def hi(a,b):
    print('%s say hi to %s'%(a,b)) 

if __name__ == '__main__':
    ob = hi('xiao','da')
    print(ob)