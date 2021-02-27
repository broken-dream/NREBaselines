import torch
import sklearn.metrics as metric
from tqdm import tqdm


class REFramework(object):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, args):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.args = args
        self.bets_metric = 0
        if torch.cuda.is_available() and self.args.use_gpu:
            self.model = model.cuda()
        else:
            self.model = model

    def train(self):
        best_metric = 0
        best_epoch = 0

        for epoch in range(self.args.epoch):
            self.model.train()
            print("------epoch {} train------".format(epoch))
            for batch_data, batch_label in tqdm(iter(self.train_loader)):
                if torch.cuda.is_available() and self.args.use_gpu:
                    for k in batch_data:
                        batch_data[k] = batch_data[k].cuda()
                    batch_label = batch_label.cuda()

                self.optimizer.zero_grad()
                logits = self.model(batch_data)
                loss = self.criterion(logits, batch_label)
                loss.backward()
                self.optimizer.step()
                print("loss:{}".format(loss))

                print("------epoch {} val------".format(epoch))
                pred = []
                truth = []
                self.model.eval()
                for batch_data, batch_label in tqdm(iter(self.val_loader)):
                    if self.args.use_gpu and torch.cuda.is_available():
                        for k in batch_data:
                            batch_data[k] = batch_data[k].cuda()
                        pred += self.model(batch_data).cpu().argmax(dim=1).numpy().tolist()
                    else:
                        pred += self.model(batch_data).argmax(dim=1).numpy().tolist()
                    truth += batch_label.numpy().tolist()
                eval_res = eval(pred, truth)
                print("acc:{}".format(eval_res["acc"]))
                print("micro_f1:{}".format(eval_res["micro_f1"]))
                print("macro_f1:{}".format(eval_res["macro_f1"]))
                print("{}:{}".format(self.args.metric, eval_res[self.args.metric]))
                if eval_res[self.args.metric] > best_metric:
                    best_metric = eval_res[self.args.metric]
                    best_epoch = epoch
                    torch.save({"state_dict": self.model.state_dict()}, self.args.save_path)
                    print("Best checkpoint")

        print("---------------------------------")
        print("bset {}:{}".format(self.args.metric, best_metric))
        print("bset res epoch:{}".format(best_epoch))
        print("---------------------------------")

    @staticmethod
    def eval(pred, truth):
        res = dict()
        res["acc"] = metric.accuracy_score(truth, pred)
        res["micro_p"] = metric.precision_score(truth, pred, average="micro")
        res["micro_r"] = metric.recall_score(truth, pred, average="micro")
        res["micro_f1"] = metric.f1_score(truth, pred, average="micro")
        res["macro_p"] = metric.precision_score(truth, pred, average="macro")
        res["macro_r"] = metric.recall_score(truth, pred, average="macro")
        res["macro_f1"] = metric.f1_score(truth, pred, average="macro")
        return res