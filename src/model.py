import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import torch.nn.functional as F


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
        self.max_sentence_size = 100
        self.pos_size = 120

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

    def forward(self, sentence, pos1, pos2):
        # 输入的bag 2D-tensor n_sample * sentence_length

        word_embedding = self.embedding(sentence).view(-1, 1, self.max_sentence_size, self.word_embedding_size)
        pos1_embedding = self.pos_embedding(pos1).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)
        pos2_embedding = self.pos_embedding(pos2).view(-1, 1, self.max_sentence_size, self.pos_embedding_size)


        input_embedding = torch.cat((word_embedding, pos1_embedding, pos2_embedding), -1)

        H = self.sentence_encoder(input_embedding).view(-1, self.kernel_num)

        A = self.attention(H)
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Out = self.classifier(M)

        return Out


if __name__ == "__main__":
    model = Encoder()
    inp1 = Variable(torch.LongTensor(np.random.randint(0, 100, (100, 1))))
    bag_sen = np.asarray([np.random.randint(0, 100, (100, 1)) for _ in range(13)])
    bag = Variable(torch.LongTensor(bag_sen)).view(13, 100)
    out = model(bag)
    print(out)
