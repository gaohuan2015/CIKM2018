import codecs

import random

import torch
import numpy as np

from torch.autograd import Variable

from gensim.models import KeyedVectors


def pos_embed(x):
    if x < -60:
        return 0
    if x >= -60 and x <= 60:
        return x + 61
    if x > 60:
        return 122


# load data from .txt file
def build_data(train_hrt_bags, word2id, relation2id, list_size=70):
    # 找出所有一个实体对不止一个关系的数据
    # for k,v in train_ht_relation.items():
    #     if len(v) > 1:
    #         print(k)
    #         print(v)

    # build text data
    bag2id = list()
    data = list()
    label = list()
    pos1 = list()
    pos2 = list()
    for key, value in train_hrt_bags.items():

        bag_text = list()
        bag_pos1 = list()
        bag_pos2 = list()
        bag2id.append(key)

        e1, e2, relation = key

        for s_instance in value:
            s_list = s_instance.strip().split()

            e1 = s_list[2]
            e2 = s_list[3]

            sentence = s_list[5:-1]

            e1_pos = sentence.index(e1)  # if e1 in s_list else -1
            e2_pos = sentence.index(e2)  # if e2 in s_list else -1

            output = []

            for i in range(list_size):
                word = word2id['BLANK']
                rel_e1 = pos_embed(i - e1_pos)
                rel_e2 = pos_embed(i - e2_pos)
                output.append([word, rel_e1, rel_e2])

            for i in range(min(list_size, len(sentence))):
                word = 0
                if sentence[i] not in word2id:
                    word = word2id['UNK']
                else:
                    word = word2id[sentence[i]]

                output[i][0] = word

            id_list = [output[i][0] for i in range(list_size)]
            e1_list = [output[i][1] for i in range(list_size)]
            e2_list = [output[i][2] for i in range(list_size)]

            # todo : 根据wikidata的命名 保证每个实体的id与mention是能够对应的
            # id_list = ([word2id[word] if word in word2id.keys() else len(word2id.keys()) - 2 for word in
            #             s_list[5:-1]] + [
            #                len(word2id.keys()) - 1] * list_size)[
            #           :list_size]
            #
            # todo :考虑一下如果句子不足最大长度 如何padding
            # e1_list = [pos_embed(i - e1_pos) for i in range(list_size)]
            # e2_list = [pos_embed(i - e2_pos) for i in range(list_size)]

            bag_text.append(id_list)
            bag_pos1.append(e1_list)
            bag_pos2.append(e2_list)

        # 出去冗余的数据（有些数据经过处理后有异常）
        if len(bag_text) <= 0:
            break

        data.append(bag_text)
        pos1.append(bag_pos1)
        pos2.append(bag_pos2)
        label.append(relation)

    # print(data_check(data, label, pos))

    return data, label, pos1, pos2, bag2id


# transform text data into torch.Tensor format
def torch_format(data, label, pos):
    data = torch.LongTensor(np.asarray(data))
    pos = torch.LongTensor(np.asarray(pos))

    return data, label, pos


# load pre-trained word embedding model in .bin or .vec format, and transform it into torch.embedding format
def load_word_embedding(path="../data/vec4.bin"):
    # word2vec
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)  # C binary format

    # print(word_vectors.similarity('woman', 'man'))

    word2id = dict()

    for word_id, word in enumerate(word_vectors.index2word):
        word2id[word] = word_id + 1

    return word2id, np.concatenate((np.random.normal(size=word_vectors.vector_size, loc=0, scale=0.05).reshape(1, -1),
                                    np.asarray(word_vectors.syn0)))


def load_word_embedding_txt(path="../data/vec.txt"):
    vec = []
    word2id = {}
    f = open(path, 'r', encoding="UTF-8")
    f.readline()
    while True:
        content = f.readline()
        if content == '':
            break
        content = content.strip().split()
        word2id[content[0]] = len(word2id)
        content = content[1:]
        content = [(float)(i) for i in content]
        vec.append(content)
    f.close()
    word2id['UNK'] = len(word2id)
    word2id['BLANK'] = len(word2id)

    dim = len(vec[0])

    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec.append(np.random.normal(size=dim, loc=0, scale=0.05))
    vec = np.array(vec, dtype=np.float32)

    np.save("../data/word2vec.npy", vec)

    return word2id, vec


# collect INTERMEDIATE data format
def data_collection(path, relation2id):
    sentences = [s for s in codecs.open(path, "r", encoding="utf8").readlines()]

    train_ht_relation = dict()  # key:(e1, e2) string value:sentence list
    train_hrt_bags = dict()  # key:(e1, e2, r) string value:sentence list
    train_head_set = dict()  # key(e) string value: tail list
    train_tail_set = dict()
    # entity_mention_map = dict()

    for s in sentences:

        s_list = s.split()

        if len(s_list) < 6:
            continue

        # 实体和实体mention的映射
        # if s_list[0] in entity_mention_map.keys():
        #     if s_list[2] != entity_mention_map[s_list[0]]:
        #         print(s_list[2], entity_mention_map[s_list[0]])
        # else:
        #     entity_mention_map[s_list[0]] = s_list[2]
        #
        # if s_list[1] in entity_mention_map.keys():
        #     if s_list[3] != entity_mention_map[s_list[1]]:
        #         print(s_list[3], entity_mention_map[s_list[1]])
        # else:
        #     entity_mention_map[s_list[1]] = s_list[3]

        e1_id = s_list[0]
        e2_id = s_list[1]

        e1_mention = s_list[2]
        e2_mention = s_list[3]

        r_mention = s_list[4]

        r_id = relation2id[r_mention] if r_mention in relation2id.keys() else relation2id["NA"]

        # fill train (h,t) dict
        if entity_mention(e1_mention, e2_mention) in train_ht_relation.keys():
            train_ht_relation[entity_mention(e1_mention, e2_mention)].add(r_id)
        else:
            train_ht_relation[entity_mention(e1_mention, e2_mention)] = set()
            train_ht_relation[entity_mention(e1_mention, e2_mention)].add(r_id)

        # fill train (h,r,t) dict
        if triplet_mention(e1_mention, e2_mention, r_id) in train_hrt_bags.keys():
            train_hrt_bags[triplet_mention(e1_mention, e2_mention, r_id)].append(s)
        else:
            train_hrt_bags[triplet_mention(e1_mention, e2_mention, r_id)] = list()
            train_hrt_bags[triplet_mention(e1_mention, e2_mention, r_id)].append(s)

        # fill head-tail dict
        if e1_mention in train_head_set.keys():
            train_head_set[e1_mention].add(e2_mention)
        else:
            train_head_set[e1_mention] = set()
            train_head_set[e1_mention].add(e2_mention)

        # fill tail-head dict
        if e2_mention in train_tail_set.keys():
            train_tail_set[e2_mention].add(e1_mention)
        else:
            train_tail_set[e2_mention] = set()
            train_tail_set[e2_mention].add(e1_mention)

    return train_ht_relation, train_hrt_bags, train_head_set, train_tail_set


def data_check(data, label, pos):
    return len(data) == len(label) == len(pos) and data_check_bag(data, pos)


def data_check_bag(data, pos):
    if len(data) == len(pos):
        for i in range(len(data)):
            if len(data[i]) != len(pos[i]):
                return False

    return True


# load relation to id file
def relation_id(path):
    relation2id = dict()
    for line in codecs.open(path, "r", encoding="utf-8").readlines():
        relation, id = line.split()
        relation2id[relation] = int(id)

    tmp = list(relation2id.items())[0][0]

    # relation2id[tmp] = 99
    # relation2id["NA"] = 0

    return relation2id


def build_path(train_ht_relation, train_hrt_bags, train_head_set, train_tail_set, bag2id):
    # build path data
    train_path = dict()
    for mention in train_ht_relation.keys():
        head, tail = entity_mention_unpack(mention)
        train_path[(head, tail)] = set()
        for tmp in train_head_set[head]:
            if tmp in train_tail_set[tail]:
                if train_ht_relation[entity_mention(head, tmp)] == {99} \
                        or train_ht_relation[entity_mention(tmp, tail)] == {99}:
                    continue

                train_path[(head, tail)].add(tmp)

    path_id = []

    entity2id = dict()

    count = 0
    for e1, e2, relation in bag2id:
        entity2id[(e1, e2)] = count
        count += 1

    with codecs.open("../data/path/path_tmp.txt", "w") as f:

        i = 0
        for e1, e2, relation in bag2id:
            line = str(i) + '\t' + str(e1) + '\t' + str(e2) + '\t' + str(relation) + '\t'
            if len(train_path[(e1, e2)]) > 0:
                line += str('\t'.join(list(train_path[(e1, e2)])))
                tmp = list(train_path[(e1, e2)])[0]
                path_id.append([entity2id[e1, tmp] if (e1, tmp) in entity2id.keys() else -1,
                                entity2id[tmp, e2] if (tmp, e2) in entity2id.keys() else -1])
            else:
                path_id.append([-1, -1])

            f.write(line)
            f.write("\n")
            i += 1

    np.save("../data/np/path.npy", np.array(path_id))

    return path_id


def entity_mention(e1, e2):
    return e1, e2


# todo:path need to redesign
def entity_mention_unpack(mention):
    l = mention
    return l[0], l[1]


def triplet_mention(e1, e2, r):
    return e1, e2, r


def triplet_mention_unpack(mention):
    l = mention
    return l[0], l[1], l[2]


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


# generate the summary report about the data set,
# todo : 生成一份关于数据集的报告，包含多少个bag，多少种关系，关系的分布，bag中句子数量的分布，word的集合等等
def report(path):
    train_ht_relation, train_hrt_bags, _, _, entity_mention_map = data_collection(path)

    return


def init():
    train_data_path = "../data/train.txt"
    test_data_path = "../data/test.txt"
    relation2id_path = "../data/relation2id.txt"

    id2, val = load_word_embedding_txt()
    relation2id = relation_id(relation2id_path)

    train_ht_relation, train_hrt_bags, train_head_set, train_tail_set = data_collection(train_data_path, relation2id)

    # f = open("../data/np/train_q&a.txt", "w", encoding="utf-8")
    #
    # i = 0
    # for key, value in train_hrt_bags.items():
    #     f.write(str(i) + '\t' + str(key[0]) + '\t' + str(key[1]) + '\t' + str(key[2]) + '\n')

    data, label, pos1, pos2, bag2id = build_data(train_hrt_bags, id2, relation2id)

    path_id = build_path(train_ht_relation, train_hrt_bags, train_head_set, train_tail_set, bag2id)

    fake_bag = [[id2["UNK"] for _ in range(70)]]
    fake_label = to_categorical([relation2id["NA"]], 100)
    fake_pos1 = [[pos_embed(i) for i in range(70)]]
    fake_pos2 = [[pos_embed(i) for i in range(70)]]

    pa_bag = []
    pa_label = []
    pa_pos1 = []
    pa_pos2 = []

    pb_bag = []
    pb_label = []
    pb_pos1 = []
    pb_pos2 = []

    for i in range(len(data)):
        p1 = path_id[i][0]
        p2 = path_id[i][1]

        if p1 == -1:
            pa_bag.append(fake_bag)
            pa_label.append(fake_label)
            pa_pos1.append(fake_pos1)
            pa_pos2.append(fake_pos2)

        else:
            pa_bag.append(data[p1])
            pa_label.append(to_categorical(label[p1], 100))
            pa_pos1.append(pos1[p1])
            pa_pos2.append(pos2[p1])

        if p2 == -1:
            pb_bag.append(fake_bag)
            pb_label.append(fake_label)
            pb_pos1.append(fake_pos1)
            pb_pos2.append(fake_pos2)

        else:
            pb_bag.append(data[p2])
            pb_label.append(to_categorical(label[p2], 100))
            pb_pos1.append(pos1[p2])
            pb_pos2.append(pos2[p2])

    np.save("../data/np/train_pa_bag.npy", pa_bag)
    np.save("../data/np/train_pa_label.npy", pa_label)
    np.save("../data/np/train_pa_pos1.npy", pa_pos1)
    np.save("../data/np/train_pa_pos2.npy", pa_pos2)

    np.save("../data/np/train_pb_bag.npy", pb_bag)
    np.save("../data/np/train_pb_label.npy", pb_label)
    np.save("../data/np/train_pb_pos1.npy", pb_pos1)
    np.save("../data/np/train_pb_pos2.npy", pb_pos2)

    np.save("../data/np/train_bag.npy", np.asarray(data))
    np.save("../data/np/train_label.npy", to_categorical(np.asarray(label)))
    np.save("../data/np/train_pos1.npy", np.asarray(pos1))
    np.save("../data/np/train_pos2.npy", np.asarray(pos2))

    _, test_hrt_bags, _, _ = data_collection(test_data_path, relation2id)

    data, label, pos1, pos2, bag2id = build_data(test_hrt_bags, id2, relation2id)

    np.save("../data/np/test_bag.npy", np.asarray(data))
    np.save("../data/np/test_label.npy", to_categorical(np.asarray(label)))
    np.save("../data/np/test_pos1.npy", np.asarray(pos1))
    np.save("../data/np/test_pos2.npy", np.asarray(pos2))


def load_all_data():
    train_bag = np.load("../data/np/train_bag.npy")
    train_label = np.load("../data/np/train_label.npy")
    train_pos1 = np.load("../data/np/train_pos1.npy")
    train_pos2 = np.load("../data/np/train_pos2.npy")

    test_bag = np.load("../data/np/test_bag.npy")
    test_label = np.load("../data/np/test_label.npy")
    test_pos1 = np.load("../data/np/test_pos1.npy")
    test_pos2 = np.load("../data/np/test_pos2.npy")

    return train_bag, train_label, train_pos1, train_pos2, test_bag, test_label, test_pos1, test_pos2


def load_train():
    train_bag = np.load("../data/np/train_bag.npy")
    train_label = np.load("../data/np/train_label.npy")
    train_pos1 = np.load("../data/np/train_pos1.npy")
    train_pos2 = np.load("../data/np/train_pos2.npy")

    # train_bag = np.load("../data/data/small_word.npy")
    # train_label = np.load("../data/data/small_y.npy")
    # train_pos1 = np.load("../data/data/small_pos1.npy")
    # train_pos2 = np.load("../data/data/small_pos2.npy")

    return train_bag, train_label, train_pos1, train_pos2


def load_train_path():
    pa_bag = np.load("../data/np/train_pa_bag.npy")
    pa_label = np.load("../data/np/train_pa_label.npy")
    pa_pos1 = np.load("../data/np/train_pa_pos1.npy")
    pa_pos2 = np.load("../data/np/train_pa_pos2.npy")

    pb_bag = np.load("../data/np/train_pb_bag.npy", )
    pb_label = np.load("../data/np/train_pb_label.npy", )
    pb_pos1 = np.load("../data/np/train_pb_pos1.npy", )
    pb_pos2 = np.load("../data/np/train_pb_pos2.npy", )

    return pa_bag, pa_label, pa_pos1, pa_pos2, pb_bag, pb_label, pb_pos1, pb_pos2


def load_test():
    test_bag = np.load("../data/np/test_bag.npy")
    test_label = np.load("../data/np/test_label.npy")
    test_pos1 = np.load("../data/np/test_pos1.npy")
    test_pos2 = np.load("../data/np/test_pos2.npy")

    # test_bag = np.load("../data/data/testall_word.npy")
    # test_label = np.load("../data/data/testall_y.npy")
    # test_pos1 = np.load("../data/data/testall_pos1.npy")
    # test_pos2 = np.load("../data/data/testall_pos2.npy")

    return test_bag, test_label, test_pos1, test_pos2


# 对数据进行采样，根据关系的数量，来选定每个关系的比例。
def data_sample(relation_num=5, NA_ratio=0.1, NA_id=-1):
    relation2id = relation_id("../data/relation2id.txt")
    train_bag, train_label, train_pos1, train_pos2, test_bag, test_label, test_pos1, test_pos2 = load_all_data()

    count = [list(train_label).count(int(v)) for _, v in relation2id.items()]

    index = sorted(range(len(count)), key=lambda k: count[k])
    index.reverse()

    sample_id = index[1:1 + relation_num]

    idx = [i for i in range(len(train_bag)) if int(train_label[i]) in sample_id]
    NA_idx = [i for i in range(len(train_bag)) if int(train_label[i]) == 99]

    slice = random.sample(NA_idx, int(len(NA_idx) * NA_ratio))

    sample_train = idx + slice

    sample_train_bag = train_bag[sample_train]
    sample_train_label = train_label[sample_train]
    sample_train_pos1 = train_pos1[sample_train]
    sample_train_pos2 = train_pos2[sample_train]

    idx = [i for i in range(len(test_bag)) if int(test_label[i]) in sample_id]
    NA_idx = [i for i in range(len(test_bag)) if int(test_label[i]) == 99]

    slice = random.sample(NA_idx, int(len(NA_idx) * NA_ratio))

    sample_test = idx + slice

    sample_test_bag = test_bag[sample_test]
    sample_test_label = test_label[sample_test]
    sample_test_pos1 = test_pos1[sample_test]
    sample_test_pos2 = test_pos2[sample_test]

    np.save("../data/sample/train_bag.npy", sample_train_bag)
    np.save("../data/sample/train_label.npy", sample_train_label)
    np.save("../data/sample/train_pos1.npy", sample_train_pos1)
    np.save("../data/sample/train_pos2.npy", sample_train_pos2)

    np.save("../data/sample/test_bag.npy", sample_test_bag)
    np.save("../data/sample/test_label.npy", sample_test_label)
    np.save("../data/sample/test_pos1.npy", sample_test_pos1)
    np.save("../data/sample/test_pos2.npy", sample_test_pos2)

    return sample_train_bag, sample_train_label, sample_train_pos1, sample_train_pos2, \
           sample_test_bag, sample_test_label, sample_test_pos1, sample_test_pos2


def shuffle(bag, label, pos1, pos2):
    index = np.array(range(len(bag)))

    np.random.shuffle(index)

    return bag[index], label[index], pos1[index], pos2[index]


if __name__ == "__main__":
    # data_sample()

    # a,b = load_word_embedding()
    init()
    print("end")
