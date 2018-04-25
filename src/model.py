import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F

from src.load import to_categorical


class Encoder(nn.Module):
    def __init__(self, word2vec):
        super(Encoder, self).__init__()

        # hyper-parameters
        self.word_embedding_size = 50
        self.sentence_representation = 50
        self.bag_size = 20
        self.windows_size = 3
        self.kernel_num = 200
        self.vocabulary_size = word2vec.shape[0]
        self.pos_embedding_size = 5
        self.relation_num = 100
        self.max_sentence_size = 70
        self.pos_size = 120
        self.hidden_dim = 200

        # embedding layer
        self.embedding = nn.Embedding(self.vocabulary_size, self.word_embedding_size)
        self.pos_embedding = nn.Embedding(self.pos_size, self.pos_embedding_size)
        # self.position2_embedding = nn.Embedding(1, self.pos_embedding_size)

        self.embedding.weight.data.copy_(torch.from_numpy(word2vec))

        # part1:sentence encoder layer
        self.sentence_encoder = nn.Sequential(
            nn.Conv2d(1, self.kernel_num, (self.word_embedding_size, self.windows_size),
                      stride=(self.word_embedding_size + 1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((1, self.word_embedding_size - self.windows_size + 1))
        )

        # part1:attention layer over bag, calculate the similarity between sentence encoder and relation embedding in KG
        self.attention = nn.Sequential(
            nn.Linear(self.kernel_num, 1),
            nn.Tanh(),
        )

        # part1:classifier layer
        self.classifier = nn.Sequential(
            nn.Linear(self.kernel_num, self.relation_num),
            # nn.ReLU()
        )

        # loss
        self.lr = nn.BCEWithLogitsLoss()

        # gru
        self.lstm = nn.GRU(self.word_embedding_size + 10, self.kernel_num, bidirectional=True, dropout=0.2,
                           batch_first=True)

        self.attention_w = Variable(torch.FloatTensor(self.kernel_num, 1), requires_grad=True).cuda()
        nn.init.xavier_uniform(self.attention_w)

        # self-define layer
        self.sen_a = Variable(torch.FloatTensor(self.kernel_num), requires_grad=True).cuda()
        nn.init.normal(self.sen_a)
        # nn.init.xavier_uniform(self.sen_a)

        self.sen_r = Variable(torch.FloatTensor(self.kernel_num, 1), requires_grad=True).cuda()
        nn.init.normal(self.sen_r)
        # nn.init.xavier_uniform(self.sen_r)

        self.relation_embedding = Variable(torch.FloatTensor(100, self.kernel_num), requires_grad=True).cuda()
        nn.init.normal(self.relation_embedding)
        # nn.init.xavier_uniform(self.relation_embedding)

        self.sen_d = Variable(torch.FloatTensor(100), requires_grad=True).cuda()
        nn.init.normal(self.sen_d)
        # nn.init.xavier_uniform(self.sen_d)

    def forward_single(self, sentence, pos1, pos2):
        # 输入的bag 2D-tensor n_sample * sentence_length

        word_embedding = self.embedding(sentence).view(-1, 1, self.max_sentence_size, self.word_embedding_size)
        pos1_embedding = self.pos_embedding(pos1).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)
        pos2_embedding = self.pos_embedding(pos2).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)

        input_embedding = torch.cat((word_embedding, pos1_embedding, pos2_embedding), -1).view(-1,
                                                                                               self.max_sentence_size,
                                                                                               self.word_embedding_size + 2 * self.pos_embedding_size)

        # H = self.sentence_encoder(input_embedding).view(-1, self.kernel_num)

        tup, _ = self.lstm(input_embedding)

        tupf = tup[:, :, range(self.hidden_dim)]
        tupb = tup[:, :, range(self.hidden_dim, self.hidden_dim * 2)]
        tup = torch.add(tupf, tupb)
        tup = tup.contiguous()

        tup1 = F.tanh(tup).view(-1, self.hidden_dim)

        tup1 = torch.matmul(tup1, self.attention_w).view(-1, 70)

        tup1 = F.softmax(tup1).view(-1, 1, 70)

        H = torch.matmul(tup1, tup).view(-1, self.hidden_dim)

        # A = self.attention(H)
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        A = F.softmax(torch.matmul(torch.mul(H, self.sen_a), self.sen_r)).view(1, -1)

        S = torch.matmul(A, H).view(-1, 1)

        Out = torch.add(torch.matmul(self.relation_embedding, S).view(-1), self.sen_d).view(1, -1)

        # Out = self.classifier(M)

        return Out

    def forward(self, sentence_bag, label_bag, pos1_bag, pos2_bag):
        label_bag = to_categorical(label_bag, 100)

        self.loss = []
        prob = []

        for i in range(len(sentence_bag)):
            outputs = self.forward_single(Variable(torch.LongTensor(sentence_bag[i])).cuda(),
                                          Variable(torch.LongTensor(pos1_bag[i])).cuda(),
                                          Variable(torch.LongTensor(pos2_bag[i])).cuda())
            prob.append(outputs.data[0])
            target = Variable(torch.FloatTensor(np.asarray(label_bag[i]).reshape((1, -1)))).cuda()

            self.loss.append(self.lr(outputs, target))

            if i == 0:
                self.total_loss = self.loss[i]
            else:
                self.total_loss += self.loss[i]

        return self.total_loss, prob

    def cal_loss(self, outputs, target):

        return torch.mean(self.lr(outputs, target))


class RNNEncoder(nn.Module):
    def __init__(self, word2vec, hidden_dim):
        self.hidden_num = hidden_dim

        # hyper-parameters
        self.word_embedding_size = 50
        self.sentence_representation = 50
        self.bag_size = 20
        self.windows_size = 3
        self.kernel_num = 200
        self.vocabulary_size = word2vec.shape[0]
        self.pos_embedding_size = 5
        self.relation_num = 100
        self.max_sentence_size = 100
        self.pos_size = 123

        self.word_embeddings = nn.Embedding(word2vec.shape[0], self.word_embedding_size)

        self.word_embeddings.weight = nn.Parameter(torch.from_numpy(word2vec))

        self.pos1_embeddings = nn.Embedding(self.pos_size, self.pos_embedding_size)
        # nn.init.xavier_uniform(self.pos1_embeddings.weight)
        nn.init.normal(self.pos1_embeddings.weight)
        self.pos2_embeddings = nn.Embedding(self.pos_size, self.pos_embedding_size)
        nn.init.normal(self.pos2_embeddings.weight)
        # nn.init.xavier_uniform(self.pos2_embeddings.weight)

        self.lstm = nn.GRU(self.word_embedding_size + 10, hidden_dim, bidirectional=True, dropout=0.2, batch_first=True)

        return

    def forward(self):



        return


if __name__ == "__main__":
    model = Encoder()
    inp1 = Variable(torch.LongTensor(np.random.randint(0, 100, (100, 1))))
    bag_sen = np.asarray([np.random.randint(0, 100, (100, 1)) for _ in range(13)])
    bag = Variable(torch.LongTensor(bag_sen)).view(13, 100)
    out = model(bag)
    print(out)
