import torchvision
from torchvision.datasets import MNIST
from PIL.Image import Image

root_path='/Users/wangjialin/projects/CMeKG_tools/torch/data/'
mnist=MNIST(root=root_path,train=True,download=False)
print(mnist)
print(mnist[0])
img=mnist[0][0]
img.show()