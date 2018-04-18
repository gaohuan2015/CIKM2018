import codecs

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

            e1_pos = s_list.index(e1) # if e1 in s_list else -1
            e2_pos = s_list.index(e2) # if e2 in s_list else -1

            # if e1_pos == -1 or e2_pos == -1:
            #     break

            e1_list = ([i - e1_pos + 3 for i in range(len(s_list))] + [len(s_list) - e1_pos + 3] * list_size)[:list_size]
            e2_list = ([i - e2_pos + 3 for i in range(len(s_list))] + [len(s_list) - e2_pos + 3] * list_size)[:list_size]

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
        relation2id[relation] = id

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


# generate the summary report about the data set,
# todo : 生成一份关于数据集的报告，包含多少个bag，多少种关系，关系的分布，bag中句子数量的分布，word的集合等等
def report(path):
    train_ht_relation, train_hrt_bags, _, _, entity_mention_map = data_collection(path)

    return


if __name__ == "__main__":
    train_data_path = "../data/train.txt"
    relation2id_path = "../data/relation2id.txt"

    _, train_hrt_bags, _, _, entity_mention_map = data_collection(train_data_path)
    id2, val = load_word_embedding()
    relation2id = relation_id(relation2id_path)

    data, label, pos1, pos2 = build_data(train_hrt_bags, entity_mention_map, id2, relation2id)

    # data, label, pos = torch_format(data, label, pos)
    print(len(data))

    print("end")
