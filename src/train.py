import datetime
import matplotlib.pyplot as plt

from itertools import chain

from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection, load_all_data, to_categorical, \
    shuffle, load_train, load_test, load_word_embedding_txt, load_train_path, load_test_path, load_train_kg, \
    load_test_kg, set_noise

from src.model import Encoder, RNN

from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, f1_score

cuda = torch.cuda.is_available()

if cuda:
    torch.cuda.manual_seed(1)


def train(word2vec, entity2vec, kg_re2vec, batch_size=50):
    train_bag, train_label, train_pos1, train_pos2, train_entity = load_train()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = load_train_path()
    kg_mid_entity, kg_path_relation = load_train_kg()

    pa_label = pa_label.reshape((-1, 100))
    pb_label = pb_label.reshape((-1, 100))
    # test_bag, test_label, test_pos1, test_pos2 = load_test()

    model = torch.load("../data/model/sentence_model_4")
    # model = RNN(word2vec, entity2vec, kg_re2vec, 100, 50)

    if cuda:
        model.cuda()
    # loss_function = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    for epoch in range(10):

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
            batch_entity = train_entity[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()
            seq_entity = Variable(torch.LongTensor(np.array(batch_entity))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_0 = torch.stack(sen_0)
            sen_0 = torch.squeeze(sen_0)
            #
            # loss_0, _ = model.s_forward(sen_0, seq_entity, y_batch=batch_label)

            # loss_0, _ = model.entity_encoder(seq_word, seq_pos1, seq_pos2, shape, seq_entity, batch_label)
            #

            # loss_0, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2. path encode

            # 2.1 path a encode

            batch_word = pa_bag[index]
            batch_pos1 = pa_pos1[index]
            batch_pos2 = pa_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_a, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_a = torch.stack(sen_a)
            sen_a = torch.squeeze(sen_a)

            # loss_a = model.s_forward(sen_a, y_batch=batch_label)
            # loss_a, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # 2.2 path b encode

            batch_word = pb_bag[index]
            batch_pos1 = pb_pos1[index]
            batch_pos2 = pb_pos2[index]

            seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
            seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
            seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

            batch_length = [len(bag) for bag in batch_word]
            shape = [0]
            for j in range(len(batch_length)):
                shape.append(shape[j] + batch_length[j])

            _, sen_b, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
            sen_b = torch.stack(sen_b)
            sen_b = torch.squeeze(sen_b)

            # loss_b, _, _ = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

            # all loss

            batch_mid_entity = mid_entity[index]
            seq_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_mid_entity]))).cuda()

            # s = [sen_a[0] + sen_b[0] for i in range(batch_size)]
            #
            # loss_path = model.s_forward(s, batch_label)

            # loss = loss_0  # + loss_a + loss_b

            batch_kg_mid_entity = kg_mid_entity[index]
            seq_kg_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_kg_mid_entity]))).cuda()

            batch_kg_relation = kg_path_relation[index]
            seq_kg_relation = Variable(torch.LongTensor(np.array([s for s in batch_kg_relation]))).cuda()

            loss, prob = model.gcn_layer(sen_0, sen_a, sen_b, seq_entity, seq_mid_entity, seq_kg_mid_entity,
                                         seq_kg_relation, batch_label)
            # loss = loss_0  # + 0.1 * loss_path

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


def eval(model, noise=1, indices=1, batch_size=10):
    if cuda:
        model.cuda()

    bag, label, pos1, pos2, entity = load_test()
    pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = load_test_path()
    kg_mid_entity, kg_path_relation = load_test_kg()

    if noise != 1:
        bag, label, pos1, pos2, entity = bag[indices], label[indices], pos1[indices], pos2[indices], entity[indices]
        pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2, mid_entity = pa_bag[indices], pa_label[
            indices], pa_pos1[indices], pa_pos2[indices], pb_bag[indices], pb_label[indices], pb_pos1[indices], pb_pos2[
                                                                                                 indices], mid_entity[
                                                                                                  indices]
        kg_mid_entity, kg_path_relation = kg_mid_entity[indices], kg_path_relation[indices]
    # model.eval()

    # print('test %d instances with %d NA relation' % (len(bag), list(label).count(99)))

    allprob = []
    alleval = []

    for i in range(int(len(bag) / batch_size)):

        batch_word = bag[i * batch_size:(i + 1) * batch_size]
        batch_label = np.asarray(label[i * batch_size:(i + 1) * batch_size])
        batch_pos1 = pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pos2[i * batch_size:(i + 1) * batch_size]
        batch_entity = entity[i * batch_size:(i + 1) * batch_size]

        seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
        seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
        seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()
        seq_entity = Variable(torch.LongTensor(np.array(batch_entity))).cuda()

        batch_length = [len(bag) for bag in batch_word]
        shape = [0]
        for j in range(len(batch_length)):
            shape.append(shape[j] + batch_length[j])

        # _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        # sen_0 = torch.stack(sen_0)
        # sen_0 = torch.squeeze(sen_0)

        # _, prob = model.s_forward(sen_0, y_batch=batch_label)
        # _, prob = model.entity_encoder(seq_entity, sen_0, batch_label)

        # _, _, prob = model(seq_word, seq_pos1, seq_pos2, shape, batch_label)

        # base : loss0
        _, sen_0, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        sen_0 = torch.stack(sen_0)
        sen_0 = torch.squeeze(sen_0)

        # path a
        batch_word = pa_bag[i * batch_size:(i + 1) * batch_size]
        batch_pos1 = pa_pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pa_pos2[i * batch_size:(i + 1) * batch_size]

        seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
        seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
        seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

        batch_length = [len(bag) for bag in batch_word]
        shape = [0]
        for j in range(len(batch_length)):
            shape.append(shape[j] + batch_length[j])

        _, sen_a, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        sen_a = torch.stack(sen_a)
        sen_a = torch.squeeze(sen_a)

        # path b
        batch_word = pb_bag[i * batch_size:(i + 1) * batch_size]
        batch_pos1 = pb_pos1[i * batch_size:(i + 1) * batch_size]
        batch_pos2 = pb_pos2[i * batch_size:(i + 1) * batch_size]

        seq_word = Variable(torch.LongTensor(np.array([s for bag in batch_word for s in bag]))).cuda()
        seq_pos1 = Variable(torch.LongTensor(np.array([s for bag in batch_pos1 for s in bag]))).cuda()
        seq_pos2 = Variable(torch.LongTensor(np.array([s for bag in batch_pos2 for s in bag]))).cuda()

        batch_length = [len(bag) for bag in batch_word]
        shape = [0]
        for j in range(len(batch_length)):
            shape.append(shape[j] + batch_length[j])

        _, sen_b, _ = model.sentence_encoder(seq_word, seq_pos1, seq_pos2, shape)
        sen_b = torch.stack(sen_b)
        sen_b = torch.squeeze(sen_b)

        batch_kg_mid_entity = kg_mid_entity[i * batch_size:(i + 1) * batch_size]
        seq_kg_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_kg_mid_entity]))).cuda()

        batch_kg_relation = kg_path_relation[i * batch_size:(i + 1) * batch_size]
        seq_kg_relation = Variable(torch.LongTensor(np.array([s for s in batch_kg_relation]))).cuda()

        batch_mid_entity = mid_entity[i * batch_size:(i + 1) * batch_size]
        seq_mid_entity = Variable(torch.LongTensor(np.array([s for s in batch_mid_entity]))).cuda()

        _, prob = model.gcn_layer(sen_0, sen_a, sen_b, seq_entity, seq_mid_entity, seq_kg_mid_entity, seq_kg_relation,
                                  batch_label)


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

    np.save("../data/test_data/eval.npy", np.array(alleval))
    np.save("../data/test_data/prob.npy", np.array(allprob))

    order = np.argsort(-allprob)

    num = 2000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    num = 4000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    num = 10000

    top100 = order[:num]
    correct_num_100 = 0.0
    for i in top100:
        if alleval[i] == 1:
            correct_num_100 += 1.0
    print(correct_num_100 / num)

    # f1 = f1_score(alleval[:len(allprob)], allprob)
    # print(f1)

    plt.plot(recall[:], precision[:], lw=2, color='navy', label='BGRU+2ATT')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.4, 1.0])
    plt.xlim([0.0, 0.5])
    plt.title('Precision-Recall Area={0:0.2f}'.format(average_precision))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig("../data/img/001.png")

    return average_precision


def test_noise(model_path):
    model = torch.load(model_path)
    s_75, s_85, s_95 = set_noise()

    eval(model, noise=0.75, indices=s_75)
    eval(model, noise=0.95, indices=s_85)
    eval(model, noise=0.95, indices=s_95)

    return


if __name__ == "__main__":
    word2vec = np.load("../data/word2vec.npy")
    kg_re2vec = np.load("../data/kg_emb/kg_re_embedding.npy")
    entity2vec = np.load("../data/kg_emb/entity_embedding.npy")

    # train(word2vec, entity2vec, kg_re2vec)

    test_noise("../data/model/epoch-gcn-20-0.665-ce")

    # model = torch.load("../data/model/sentence_model_0")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_1")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_2")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_3")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_4")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_5")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_6")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_7")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_8")
    # eval(model)
    # model = torch.load("../data/model/sentence_model_9")
    # eval(model)
