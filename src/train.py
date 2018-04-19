import datetime

from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection
from src.model import Encoder

from torch.autograd import Variable

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec, train_bag, train_label, train_pos1, train_pos2):
    model = Encoder(word2vec)
    if cuda:
        model.cuda()
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), weight_decay=10e-5)

    for epoch in range(25):

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

        for i in range(len(train_bag)):
            optimizer.zero_grad()

            outputs = model(Variable(torch.LongTensor(train_bag[i])).cuda(),
                            Variable(torch.LongTensor(train_pos1[i])).cuda(),
                            Variable(torch.LongTensor(train_pos2[i])).cuda())
            target = Variable(torch.LongTensor(np.asarray(train_label[i]).reshape((1)))).cuda()
            loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                starttime = endtime


if __name__ == "__main__":
    train_data_path = "../data/train.txt"
    relation2id_path = "../data/relation2id.txt"

    _, train_hrt_bags, _, _, entity_mention_map = data_collection(train_data_path)
    id2, val = load_word_embedding()
    relation2id = relation_id(relation2id_path)

    data, label, pos1, pos2 = build_data(train_hrt_bags, entity_mention_map, id2, relation2id)
    # data, label, pos = load.torch_format(data, label, pos)

    # model = Encoder(val)

    train(val, data, label, pos1, pos2)
