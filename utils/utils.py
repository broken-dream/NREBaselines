import json


def load_vocab(word2id_path, unk_token, pad_token):
    f = open(word2id_path, "r")
    word2id = json.load(f)
    if unk_token not in word2id:
        word2id[unk_token] = len(word2id)
    if pad_token not in word2id:
        word2id[pad_token] = len(word2id)
    f.close()
    return word2id


def load_rel2id(rel2id_path):
    f = open(rel2id_path, "r")
    rel2id = json.load(f)
    f.close()
    return rel2id


def load_raw_data(raw_data_path):
    res = []
    with open(raw_data_path) as f:
        for line in f:
            res.append(json.loads(line))

    return res