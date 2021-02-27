from .utils import load_vocab
import numpy as np


class BasicTokenizer(object):
    def __init__(self, word2id_path, max_length=128, unk_token="[UNK]", pad_token="[PAD]"):
        self.word2id = load_vocab(word2id_path, unk_token, pad_token)
        self.max_length = max_length
        self.unk_token = unk_token
        self.pad_token = pad_token

    def basic_tokenize(self, tokens, pos_head, pos_tail):
        indexed_tokens = []
        for token in tokens:
            token = token.lower()
            if token in self.word2id:
                indexed_tokens.append(self.word2id[token])
            else:
                indexed_tokens.append(self.word2id[self.unk_token])

        # get mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(indexed_tokens)] = 1

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(self.word2id['[PAD]'])

        # get pos information
        posvec_head = np.zeros(self.max_length, dtype=np.int32)
        posvec_tail = np.zeros(self.max_length, dtype=np.int32)
        head_in_index = min(self.max_length, pos_head[0])
        tail_in_index = min(self.max_length, pos_tail[0])

        for i in range(self.max_length):
            posvec_head[i] = i - head_in_index + self.max_length
            posvec_tail[i] = i - tail_in_index + self.max_length

        return [indexed_tokens, posvec_head, posvec_tail, mask]


class GloveTokenizer(BasicTokenizer):
    def __init__(self, word2id_path, max_length=128, unk_token="[UNK]", pad_token="[PAD]"):
        super().__init__(word2id_path, max_length=max_length, unk_token=unk_token, pad_token=pad_token)

    def tokenize(self, tokens, pos_head, pos_tail):
        return self.basic_tokenize(tokens, pos_head, pos_tail)