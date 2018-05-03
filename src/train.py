import datetime
import matplotlib.pyplot as plt

from itertools import chain

from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection, load_all_data, to_categorical, \
    shuffle, load_train, load_test, load_word_embedding_txt, load_train_path

from src.model import Encoder, RNN

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec, batch_size=50):
    train_bag, train_label, train_pos1, train_pos2 = load_train()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2 = load_train_path()

    pa_label = pa_label.reshape((-1, 100))
    pb_label = pb_label.reshape((-1, 100))
    # test_bag, test_label, test_pos1, test_pos2 = load_test()

    # model = torch.load("../data/model/sentence_model_4")
    model = RNN(len(word2vec[0]), 200, len(word2vec), word2vec, 50)

    if cuda:
        model.cuda()
    # loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(5):

        temp_order = list(range(len(train_bag)))

        np.random.shuffle(temp_order)

        # train_bag, train_label, train_pos1, train_pos2 = shuffle(train_bag, train_label, train_pos1, train_pos2)

        running_loss = 0.0
        print("每个epoch需要多个instance" + str(len(train_bag)))

        starttime = datetime.datetime.now()

        for i in range(int(len(train_bag) / batch_size)):

            optimizer.zero_grad()

            # 1. direct sentence encode
            index = temp_order[i * batch_size:(i + 1) * batch_size]

            batch_word = train_bag[index]
            batch_label = train_label[index]
            batch_pos1 = train_pos1[index]
            batch_pos2 = train_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            # seq_label = np.array([s for bag in batch_label for s in bag])
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            loss_0, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2. path encode

            # 2.1 path a encode

            batch_word = pa_bag[index]
            batch_label = pa_label[index]
            batch_pos1 = pa_pos1[index]
            batch_pos2 = pa_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            sen_a, _, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2)

            loss_a, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2.2 path b encode

            batch_word = pb_bag[index]
            batch_label = pb_label[index]
            batch_pos1 = pb_pos1[index]
            batch_pos2 = pb_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            loss_b, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # all loss

            loss = loss_0 + loss_a + loss_b

            # target = Variable(torch.LongTensor(np.asarray(train_label[i]).reshape((1)))).cuda()
            # loss = loss_function(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

                endtime = datetime.datetime.now()
                print((endtime - starttime).seconds)
                starttime = endtime

                # eval(model, test_bag, test_label, test_pos1, test_pos2)

                # torch.save(model, "../data/model/sentence_model_%s" % (str(i)))

        torch.save(model, "../data/model/sentence_model_%s" % (str(epoch)))


def eval(model, bag, label, pos1, pos2, batch_size=10):
    if cuda:
        model.cuda()

    # model.eval()

    # print('test %d instances with %d NA relation' % (len(bag), list(label).count(99)))

    allprob = []
    alleval = []

    for i in range(int(len(bag) / batch_size)):

        batch_word = bag[i * batch_size:(i + 1) * batch_size]
        batch_label = np.asarray(label[i * batch_size:(i + 1) * batch_size])
        batch_pos1 = pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pos2[i * batch_size:(i + 1) * batch_size]

        seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
        # seq_label = np.array([s for bag in batch_label for s in bag])
        seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
        seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

        batch_length = [len(bag) for bag in batch_word]
        shape = [0]
        for j in range(len(batch_length)):
            shape.append(shape[j] + batch_length[j])

        _, _, prob = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)
        print(i)

        for single_prob in prob:
            allprob.append(single_prob[0:99])
        for single_eval in batch_label:
            alleval.append(single_eval[0:99])

    allprob = np.reshape(np.array(allprob), (-1))
    alleval = np.reshape(np.array(alleval), (-1))

    allprob = np.asarray(allprob)
    # acc = accuracy_score(np.argmax(allprob, 1), label[:len(allprob)])
    # print('test accuracy ' + str(acc))

    # reshape prob matrix
    # allprob = np.reshape(allprob[:, 1:100], (-1))
    # eval_y = np.reshape(to_categorical(label)[:, 1:100], (-1))

    # order = np.argsort(-prob)

    precision, recall, threshold = precision_recall_curve(alleval[:len(allprob)], allprob)
    average_precision = average_precision_score(alleval[:len(allprob)], allprob)
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
    word2vec = np.load("../data/word2vec.npy")

    # train(word2vec)
    test_bag, test_label, test_pos1, test_pos2 = load_test()

    model = torch.load("../data/model/sentence_model_0")
    eval(model, test_bag, test_label, test_pos1, test_pos2)
