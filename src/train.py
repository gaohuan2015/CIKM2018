import datetime
import matplotlib.pyplot as plt

from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection, load_all_data, to_categorical, \
    shuffle
from src.model import Encoder

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score

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

        train_bag, train_label, train_pos1, train_pos2 = shuffle(train_bag, train_label, train_pos1, train_pos2)

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

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

        eval(model, test_bag, test_label, test_pos1, test_pos2)

    torch.save(model, "../data/model/sentence_model")


def eval(model, bag, label, pos1, pos2):
    if cuda:
        model.cuda()

    model.eval()

    print('test %d instances with %d NA relation' % (len(bag), list(label).count(99)))

    prob = []
    for i in range(len(bag)):
        outputs = model(Variable(torch.LongTensor(bag[i])).cuda(),
                        Variable(torch.LongTensor(pos1[i])).cuda(),
                        Variable(torch.LongTensor(pos2[i])).cuda())

        prob.append(outputs.data[0].cpu().numpy())

    prob = np.asarray(prob)
    acc = accuracy_score(np.argmax(prob, 1), label)
    print('test accuracy ' + str(acc))

    # reshape prob matrix
    prob = np.reshape(prob[:, :99], (-1))
    eval_y = np.reshape(to_categorical(label)[:, :99], (-1))

    # order = np.argsort(-prob)

    precision, recall, threshold = precision_recall_curve(eval_y, prob)
    average_precision = average_precision_score(eval_y, prob)
    print('test average precision' + str(average_precision))

    plt.plot(recall[:], precision[:], lw=2, color='navy', label='BGRU+2ATT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 0.4])
    plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig("../data/img/001.png")

    return prob


if __name__ == "__main__":
    id2, val = load_word_embedding()

    train(val)

    # model = torch.load("../data/model/sentence_model")
    # eval(model, test_bag, test_label, test_pos1, test_pos2)
