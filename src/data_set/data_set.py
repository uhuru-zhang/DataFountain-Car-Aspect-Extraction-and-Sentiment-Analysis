import torch.utils.data as D
import json
from torch.utils.data import Dataset


class DatafountainDataSet(Dataset):

    def __init__(self, train=True):
        json_file = "./data_set/train.json"

        dict_datas = json.loads(open(json_file).readline())
        indexes = list(sorted([index for index in dict_datas]))

        seg_index = int(len(indexes) * 0.8)

        if train:
            # 单个元素必须也得是 list
            self.datas = [
                (key, dict_datas[key]["content_indexes"], dict_datas[key]["sentiment_values"])
                for key in indexes[: seg_index]]
        else:
            self.datas = [
                (key, dict_datas[key]["content_indexes"], dict_datas[key]["sentiment_values"])
                for key in indexes[seg_index:]]

        print(json_file, train, len(self.datas))

    def __getitem__(self, index):
        return self.datas[index]

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    train_loader = D.DataLoader(TextDataSet(train=True),
                                batch_size=2, shuffle=False, num_workers=1
                                )

    for i, data in enumerate(train_loader):
        print(data)
