import datetime

from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection, load_all_data
from src.model import Encoder

from torch.autograd import Variable
from sklearn.metrics import accuracy_score

cuda = torch.cuda.is_available()

train_bag, train_label, train_pos1, train_pos2, test_bag, test_label, test_pos1, test_pos2 = load_all_data()

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec):
    model = Encoder(word2vec)


    if cuda:
        model.cuda()
    loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, )

    for epoch in range(10):

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

        eval(model)

        for i in range(len(train_bag)):

            outputs = model(Variable(torch.LongTensor(train_bag[i])).cuda(),
                            Variable(torch.LongTensor(train_pos1[i])).cuda(),
                            Variable(torch.LongTensor(train_pos2[i])).cuda())
            target = Variable(torch.LongTensor(np.asarray(train_label[i]).reshape((1)))).cuda()
            loss = loss_function(outputs, target)
            loss.backward()
            # optimizer.step()

            running_loss += loss.data[0]

            if (i + 1) % 50 == 0:
                optimizer.step()
                optimizer.zero_grad()

            if i % 20000 == 19999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                starttime = endtime

    torch.save(model, "../data/model/sentence_model")


def eval(model):
    if cuda:
        model.cuda()

    model.eval()

    prob = []
    for i in range(len(test_bag)):
        outputs = model(Variable(torch.LongTensor(test_bag[i])).cuda(),
                        Variable(torch.LongTensor(test_pos1[i])).cuda(),
                        Variable(torch.LongTensor(test_pos2[i])).cuda())

        prob.append(outputs.data[0].cpu().numpy())

    prob = np.asarray(prob)
    acc = accuracy_score(np.argmax(prob, 1), test_label)
    print(acc)

    return prob


if __name__ == "__main__":

    id2, val = load_word_embedding()

    train(val)
