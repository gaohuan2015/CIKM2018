import codecs

import random

import torch
import numpy as np

from torch.autograd import Variable

from gensim.models import KeyedVectors


# load data from .txt file
def build_data(train_hrt_bags, entity_mention_map, word2id, relation2id, list_size=100):
    # 找出所有一个实体对不止一个关系的数据
    # for k,v in train_ht_relation.items():
    #     if len(v) > 1:
    #         print(k)
    #         print(v)

    # build text data
    data = list()
    label = list()
    pos1 = list()
    pos2 = list()
    for key, value in train_hrt_bags.items():

        bag_text = list()
        bag_pos1 = list()
        bag_pos2 = list()

        e1, e2, relation = triplet_mention_unpack(key)

        for s_instance in value:
            s_list = s_instance.split()

            e1 = s_list[2]
            e2 = s_list[3]

            # todo : 根据wikidata的命名 保证每个实体的id与mention是能够对应的
            id_list = ([word2id[word] if word in word2id.keys() else 0 for word in s_list[5:]] + [0] * list_size)[
                      :list_size]

            e1_pos = s_list.index(e1)  # if e1 in s_list else -1
            e2_pos = s_list.index(e2)  # if e2 in s_list else -1

            # if e1_pos == -1 or e2_pos == -1:
            #     break

            e1_list = ([i - e1_pos + 3 for i in range(len(s_list))] + [len(s_list) - e1_pos + 3] * list_size)[
                      :list_size]
            e2_list = ([i - e2_pos + 3 for i in range(len(s_list))] + [len(s_list) - e2_pos + 3] * list_size)[
                      :list_size]

            bag_text.append(id_list)
            bag_pos1.append(e1_list)
            bag_pos2.append(e2_list)

        # 出去冗余的数据（有些数据经过处理后有异常）
        if len(bag_text) <= 0:
            break

        data.append(bag_text)
        pos1.append(bag_pos1)
        pos2.append(bag_pos2)
        label.append(relation2id[relation])

    # print(data_check(data, label, pos))

    return data, label, pos1, pos2


# transform text data into torch.Tensor format
def torch_format(data, label, pos):
    data = torch.LongTensor(np.asarray(data))
    pos = torch.LongTensor(np.asarray(pos))

    return data, label, pos


# load pre-trained word embedding model in .bin or .vec format, and transform it into torch.embedding format
def load_word_embedding(path="../data/vec4.bin"):
    # word2vec
    word_vectors = KeyedVectors.load_word2vec_format(path, binary=True)  # C binary format

    print(word_vectors.similarity('woman', 'man'))

    word2id = dict()

    for word_id, word in enumerate(word_vectors.index2word):
        word2id[word] = word_id + 1

    return word2id, np.concatenate((np.zeros(word_vectors.vector_size).reshape(1, -1), np.asarray(word_vectors.syn0)))


# collect INTERMEDIATE data format
def data_collection(path):
    sentences = [s for s in codecs.open(path, "r", encoding="utf8").readlines()]

    train_ht_relation = dict()  # key:(e1, e2) string value:sentence list
    train_hrt_bags = dict()  # key:(e1, e2, r) string value:sentence list
    train_head_set = dict()  # key(e) string value: tail list
    train_tail_set = dict()
    entity_mention_map = dict()

    for s in sentences:
        s_list = s.split()
        if len(s_list) < 6:
            continue

        # 实体和实体mention的映射
        if s_list[0] in entity_mention_map.keys():
            if s_list[2] != entity_mention_map[s_list[0]]:
                print(s_list[2], entity_mention_map[s_list[0]])
        else:
            entity_mention_map[s_list[0]] = s_list[2]

        if s_list[1] in entity_mention_map.keys():
            if s_list[3] != entity_mention_map[s_list[1]]:
                print(s_list[3], entity_mention_map[s_list[1]])
        else:
            entity_mention_map[s_list[1]] = s_list[3]

        # fill train (h,t) dict
        if entity_mention(s_list[0], s_list[1]) in train_ht_relation.keys():
            train_ht_relation[entity_mention(s_list[0], s_list[1])].add(s_list[4])
        else:
            train_ht_relation[entity_mention(s_list[0], s_list[1])] = set()
            train_ht_relation[entity_mention(s_list[0], s_list[1])].add(s_list[4])

        # fill train (h,r,t) dict
        if triplet_mention(s_list[0], s_list[1], s_list[4]) in train_hrt_bags.keys():
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])].append(s)
        else:
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])] = list()
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])].append(s)

        # fill head-tail dict
        if s_list[0] in train_head_set.keys():
            train_head_set[s_list[0]].add(s_list[1])
        else:
            train_head_set[s_list[0]] = set()
            train_head_set[s_list[0]].add(s_list[1])

        # fill tail-head dict
        if s_list[1] in train_tail_set.keys():
            train_tail_set[s_list[1]].add(s_list[0])
        else:
            train_tail_set[s_list[1]] = set()
            train_tail_set[s_list[1]].add(s_list[0])

    return train_ht_relation, train_hrt_bags, train_head_set, train_tail_set, entity_mention_map


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

    relation2id[tmp] = 99
    relation2id["NA"] = 0

    return relation2id


def build_path(train_ht_relation, train_hrt_bags, entity_mention_map, train_head_set, train_tail_set):
    # build path data
    train_path = list()
    for mention in train_ht_relation.keys():
        head, tail = entity_mention_unpack(mention)
        for tmp in train_head_set[head]:
            if tmp in train_tail_set[tail]:
                if train_ht_relation[entity_mention(head, tmp)] == {"NA"} \
                        or train_ht_relation[entity_mention(tmp, tail)] == {"NA"}:
                    continue
                train_path.append([head, tmp, tail])

    with codecs.open("../data/path_tmp.txt", "w") as f:
        for head, tmp, tail in train_path:
            if train_ht_relation[entity_mention(head, tail)] == {"NA"}:
                continue
            print(entity_mention_map[head], entity_mention_map[tmp], entity_mention_map[tail])
            print(train_ht_relation[entity_mention(head, tail)])
            print(train_ht_relation[entity_mention(head, tmp)])
            print(train_ht_relation[entity_mention(tmp, tail)])
            ra = list(train_ht_relation[entity_mention(head, tmp)])[0]
            rb = list(train_ht_relation[entity_mention(tmp, tail)])[0]
            for s in train_hrt_bags[triplet_mention(head, tmp, ra)]:
                print(s)
            for s in train_hrt_bags[triplet_mention(tmp, tail, rb)]:
                print(s)

    return train_ht_relation


def entity_mention(e1, e2):
    return e1 + " " + e2


def entity_mention_unpack(mention):
    l = mention.split()
    return l[0], l[1]


def triplet_mention(e1, e2, r):
    return e1 + " " + e2 + " " + r


def triplet_mention_unpack(mention):
    l = mention.split()
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

    id2, val = load_word_embedding()
    relation2id = relation_id(relation2id_path)

    _, train_hrt_bags, _, _, entity_mention_map = data_collection(train_data_path)
    data, label, pos1, pos2 = build_data(train_hrt_bags, entity_mention_map, id2, relation2id)

    np.save("../data/np/train_bag.npy", np.asarray(data))
    np.save("../data/np/train_label.npy", np.asarray(label))
    np.save("../data/np/train_pos1.npy", np.asarray(pos1))
    np.save("../data/np/train_pos2.npy", np.asarray(pos2))

    # data, label, pos = torch_format(data, label, pos)

    _, test_hrt_bags, _, _, entity_mention_map = data_collection(test_data_path)
    data, label, pos1, pos2 = build_data(test_hrt_bags, entity_mention_map, id2, relation2id)

    np.save("../data/np/test_bag.npy", np.asarray(data))
    np.save("../data/np/test_label.npy", np.asarray(label))
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
    index = np.array(len(bag))

    np.shuffle(index)

    return bag[index], label[index], pos1[index], pos2[index]


if __name__ == "__main__":
    # data_sample()

    relation_id("../data/relation2id.txt")
    print("end")
