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
from sklearn.metrics import precision_recall_curve, average_precision_score
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
from src.load import load_word_embedding_txt, to_categorical, relation_id, pos_embed
from src.model import RNN

# relation2id_path = "../data/relation2id.txt"
#
# id2, word2vec = load_word_embedding_txt()
# relation2id = relation_id(relation2id_path)
#
# fake_bag = [[id2["ZERO"] for _ in range(70)]]
# fake_label = to_categorical([relation2id["NA"]], 100)
# fake_pos1 = [[pos_embed(i) for i in range(70)]]
# fake_pos2 = [[pos_embed(i) for i in range(70)]]
#
# batch_word = [fake_bag, fake_bag]
# batch_pos1 = [fake_pos1, fake_pos1]
# batch_pos2 = [fake_pos2, fake_pos1]
#
#
# seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).view(-1, 70).cuda()
# seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).view(-1, 70).cuda()
# seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).view(-1, 70).cuda()
#
# batch_length = [len(bag) for bag in batch_word]
# shape = [0]
# for j in range(len(batch_length)):
#     shape.append(shape[j] + batch_length[j])
#
# word_embeddings = nn.Embedding(word2vec.shape[0], word2vec.shape[1])
# word_embeddings.weight = nn.Parameter(torch.from_numpy(word2vec))
#
#
# rnn = RNN(len(word2vec[0]), 200, len(word2vec), word2vec, 50)
# rnn.cuda()
# rnn.train()
#
# s = rnn.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
#
# print("end")

allprob = np.load("../data/test_data/prob.npy")
alleval = np.load("../data/test_data/eval.npy")

precision, recall, threshold = precision_recall_curve(alleval[:len(allprob)], allprob)
average_precision = average_precision_score(alleval[:len(allprob)], allprob)
print('test average precision' + str(average_precision))

a = np.load("../data/path/kg_path_relation.npy")
b = np.load("../data/path/kg_mid_entity.npy")

print(0)
