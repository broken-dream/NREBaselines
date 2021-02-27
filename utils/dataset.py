import torch
import torch.utils.data as data
from .utils import load_rel2id, load_raw_data


class SentenceDataset(data.Dataset):
    def __init__(self, raw_data_path, rel2id_path, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.rel2id = load_rel2id(rel2id_path)
        self.data = []
        raw_data = load_raw_data(raw_data_path)
        for item in raw_data:
            tmp = self.tokenizer.tokenize(item["token"], item["h"]["pos"], item["t"]["pos"])
            tmp.append(self.rel2id[item["relation"]])
            self.data.append(tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate_fn(data):
        batch_train = {"token": [],
                       "h_pos": [],
                       "t_pos": [],
                       "mask": []}
        batch_label = []

        for ins in data:
            batch_train["token"].append(ins[0])
            batch_train["h_pos"].append(ins[1])
            batch_train["t_pos"].append(ins[2])
            batch_train["mask"].append(ins[3])

        for k in batch_train:
            batch_train[k] = torch.tensor(batch_train[k]).long()

        batch_label = torch.tensor(batch_label).long()

        return batch_train, batch_label


def get_loader(dataset, batch_size, shuffle, num_workers=8):
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  collate_fn=dataset.collate_fn,
                                  pin_memory=True)
    return data_loader
