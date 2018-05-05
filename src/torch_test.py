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
from gensim.models import word2vec
from torch.autograd import Variable

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.autograd as autograd
# import numpy as np
#
# # 预测值f(x) 构造样本，神经网络输出层
# inputs_tensor = torch.FloatTensor([
#     [10, 2, 1, -2, -3],
#     [-1, -6, -0, -3, -5],
#     [-5, 4, 8, 2, 1]
# ])
#
# # 真值y
# targets_tensor = torch.LongTensor([1, 3, 2])
# # targets_tensor = torch.LongTensor([1])
#
# inputs_variable = autograd.Variable(inputs_tensor, requires_grad=True)
# targets_variable = autograd.Variable(targets_tensor)
# print('input tensor(nBatch x nClasses): {}'.format(inputs_tensor.shape))
# print('target tensor shape: {}'.format(targets_tensor.shape))
#
# loss = nn.CrossEntropyLoss()
# output = loss(inputs_variable, targets_variable)
# # output.backward()
# print('pytorch 内部实现的CrossEntropyLoss: {}'.format(output))


#
# a = torch.ones(25, 300)
# b = torch.ones(22, 300)
# c = torch.ones(15, 300)
# pad_sequence([a, b, c]).size()
# torch.Size([25, 3, 300])
from src.load import load_word_embedding_txt
from src.model import RNN

id2, word2vec = load_word_embedding_txt()

fake_bag = [[id2["ZERO"] for _ in range(70)]]

rnn = RNN(len(word2vec[0]), 200, len(word2vec), word2vec, 50)

print("end")