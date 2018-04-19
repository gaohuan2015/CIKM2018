import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

# torch.manual_seed(1)
#
# word_to_ix = {"hello": 0, "world": 1}
# embeds = nn.Embedding(3, 5)  # 2 words in vocab, 5 dimensional embeddings
# lookup_tensor = torch.LongTensor([[0, 1], [1,1]])
# hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

# 预测值f(x) 构造样本，神经网络输出层
inputs_tensor = torch.FloatTensor( [
 [10, 2, 1,-2,-3],
 [-1,-6,-0,-3,-5],
 [-5, 4, 8, 2, 1]
 ])

# 真值y
targets_tensor = torch.LongTensor([1,3,2])
# targets_tensor = torch.LongTensor([1])

inputs_variable = autograd.Variable(inputs_tensor, requires_grad=True)
targets_variable = autograd.Variable(targets_tensor)
print('input tensor(nBatch x nClasses): {}'.format(inputs_tensor.shape))
print('target tensor shape: {}'.format(targets_tensor.shape))

loss = nn.CrossEntropyLoss()
output = loss(inputs_variable, targets_variable)
# output.backward()
print('pytorch 内部实现的CrossEntropyLoss: {}'.format(output))
