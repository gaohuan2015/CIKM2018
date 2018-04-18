from src import load

import torch
import numpy as np
import torch.optim as optim

from src.load import load_word_embedding, relation_id, build_data, data_collection
from src.model import Encoder

from torch.autograd import Variable


def train(train_bag, train_pos, ):
    model = Encoder()
    loss_function = torch.nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), weight_decay=10e-5)

    for epoch in range(25):
        for sentence, tags in train_bag:
            model.zero_grad()

    optimizer.zero_grad()


if __name__ == "__main__":
    train_data_path = "../data/train.txt"
    relation2id_path = "../data/relation2id.txt"

    _, train_hrt_bags, _, _, entity_mention_map = data_collection(train_data_path)
    id2, val = load_word_embedding()
    relation2id = relation_id(relation2id_path)

    data, label, pos1, pos2 = build_data(train_hrt_bags, entity_mention_map, id2, relation2id)
    # data, label, pos = load.torch_format(data, label, pos)

    model = Encoder(val)

    c = model(Variable(torch.LongTensor(np.asarray(data[0]))), Variable(torch.LongTensor(np.asarray(pos1[0]))),
              Variable(torch.LongTensor(np.asarray(pos2[0]))))

    print(c)
