import codecs


# from gensim.models import KeyedVectors


def init():
    # word2vec
    # word_vectors = KeyedVectors.load_word2vec_format('../data/vec4.bin', binary=True)  # C binary format
    #
    # print(word_vectors.similarity('woman', 'man'))

    sentences = [s for s in codecs.open("../data/test.txt", "r", encoding="utf8").readlines()]

    print(len(sentences))
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
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])].append(' '.join(s_list[4:]))
        else:
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])] = list()
            train_hrt_bags[triplet_mention(s_list[0], s_list[1], s_list[4])].append(' '.join(s_list[4:]))

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

    # 找出所有一个实体对不止一个关系的数据
    # for k,v in train_ht_relation.items():
    #     if len(v) > 1:
    #         print(k)
    #         print(v)

    # build path
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


if __name__ == "__main__":
    init()
