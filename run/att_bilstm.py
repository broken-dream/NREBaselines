import torch
import torch.nn as nn
import torch.optim as optim
from model.bilstm_att import BilitmAtt
from utils.dataset import SentenceDataset, get_loader
from module.framwork import REFramework
from utils.tokenizer import GloveTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    # file path
    parser.add_argument("--train_path", default="dataset/scierc-ea-gnn/scierc_ea_gnn_train.json",
                        help="train file path")
    parser.add_argument("--val_path", default="dataset/scierc-ea-gnn/scierc_ea_gnn_val.json",
                        help="val file path")
    parser.add_argument("--test_path", default="dataset/scierc-ea-gnn/scierc_ea_gnn_test.json",
                        help="test file path")
    parser.add_argument("--save_path", default="dataset/checkpoint.pth.tar",
                        help="save path")
    parser.add_argument("--word2id_path", default="dataset/checkpoint.pth.tar",
                        help="word -> id map path")
    parser.add_argument("--word_emb_path", default="dataset/checkpoint.pth.tar",
                        help="word embedding path")
    parser.add_argument("--rel2id_path", default="dataset/checkpoint.pth.tar",
                        help="relation type -> id map path")

    # hyper parameters
    parser.add_argument("--lr", default=1, type=float,
                        help="learning rate")
    parser.add_argument("--word_emb_size", default=300, type=int,
                        help="word embedding size")
    parser.add_argument("--pos_emb_size", default=50, type=int,
                        help="position embedding size")
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="lstm output size")
    parser.add_argument("--class_num", default=7, type=int,
                        help="class number")
    parser.add_argument("--max_length", default=128, type=int,
                        help="max sequence length")
    parser.add_argument("--num_layers", default=1, type=int,
                        help="lstm layers")
    parser.add_argument("--bidirectional", default=True, type=bool,
                        help="bidirectional lstm or not")
    parser.add_argument("--emb_dropout", default=0.5, type=float,
                        help="word embedding dropout ratio")
    parser.add_argument("--emb_dropout", default=0.5, type=float,
                        help="word embedding dropout ratio")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="lstm output dropout ratio")
    parser.add_argument("--linear_dropout", default=0.5, type=float,
                        help="attention dropout ratio")
    parser.add_argument('--L2_decay', type=float, default=1e-5,
                        help='L2 weight decay')

    # others
    parser.add_argument("--batch_size", default=20, type=int,
                        help="batch size")
    parser.add_argument("--epoch", default=50, type=int,
                        help="num of train epoch")
    parser.add_argument("--use_gpu", default=True, type=bool,
                        help="use gpu or note")

    parser.add_argument("--mode", default="train",
                        help="train or test")

    args = parser.parse_args()

    tokenizer = GloveTokenizer(word2id_path=args.word2id_path,
                               max_length=args.max_length)

    model = BilitmAtt(args)
    if args.use_gpu:
        model = model.cuda()

    optimizer = optim.Adadelta(model.parameters,
                               lr=args.lr,
                               weight_decay=args.L2_decay)
    criterion = nn.CrossEntropyLoss()

    if args.mode == "train":
        train_dataset = SentenceDataset(raw_data_path=args.train_path,
                                        rel2id_path=args.rel2id_path,
                                        tokenizer=tokenizer)
        train_loader = get_loader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True)

        val_dataset = SentenceDataset(raw_data_path=args.val_path,
                                      rel2id_path=args.rel2id_path,
                                      tokenizer=tokenizer)
        val_loader = get_loader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                shuffle=True)

        framework = REFramework(model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                optimizer=optimizer,
                                criterion=criterion,
                                args=args)

        framework.train()
        print("finish training")
    else:
        test_dataset = SentenceDataset(raw_data_path=args.test_path,
                                       rel2id_path=args.rel2id_path,
                                       tokenizer=tokenizer)
        test_loader = get_loader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True)
        model.load_state_dict(torch.load(args.save_path)["state_dict"])
        model.eval()
        for batch_data, batch_label in tqdm(iter(test_loader)):
            if self.args.use_gpu and torch.cuda.is_available():
                for k in batch_data:
                    batch_data[k] = batch_data[k].cuda()
                pred += self.model(batch_data).cpu().argmax(dim=1).numpy().tolist()
            else:
                pred += self.model(batch_data).argmax(dim=1).numpy().tolist()
            truth += batch_label.numpy().tolist()

        eval_res = model.eval(pred, truth)
        for key in eval_res:
            print("test {}:{}".format(key, eval_res[key]))


