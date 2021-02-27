import torch
import torch.nn as nn
from utils.embedding import Embedding


class BilitmAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding = Embedding(word_emb_path=args.word_emb_path,
                                   max_length=args.max_length,
                                   word_emb_size=args.word_emb_size,
                                   pos_emb_size=args.pos_emb_size)

        self.bilstm = nn.LSTM(
            input_size=args.word_emb_size+args.pos_emb_size*2,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional
        )

        self.tanh = nn.Tanh()
        self.emb_dropout = nn.Dropout(args.emb_dropout)
        self.lstm_dropout = nn.Dropout(args.lstm_dropout)
        self.linear_dropout = nn.Dropout(args.linear_dropout)

        self.linear = nn.Linear(in_features=args.hidden_size,
                                out_features=args.class_num,
                                bias=True)

        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0)

        self.att_vec = nn.Parameter(torch.randn(1, 1, args.hidden_size))

    def bilstm_layer(self, x):
        x = x.transpose(0, 1) # B*emb_size*L
        x, hc = self.bilstm(x)
        x = x.transpose(0, 1) # B*L*2H
        x = x.view(-1, self.args.max_length, 2, self.args.hidden_size)
        x = torch.sum(x, dim=2) # B*L*H
        return x

    def att_layer(self, x, mask):
        att_vec = self.att_vec.expand(self.args.batch_size, self.args.max_length, -1) # B*H*1
        att_score = torch.bmm(self.tanh(x), att_vec).sequeeze(-1)

        att_score = att_score.mask_fill(mask.eq(0), float('-inf'))
        att_weight = nn.functional.softmax(att_score)

        res = torch.bmm(x.transpose(1, 2), att_weight)
        res = self.tanh(res)

        return res

    def forward(self, data):
        x = self.embedding(data)
        x = self.emb_dropout(x)

        x = self.bilstm_layer(x)
        x = self.lstm_dropout(x)

        x = self.att_layer(x, data["mask"])
        x = self.linear_dropout(x)

        logits = self.linear(x)
        return logits

