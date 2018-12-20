import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

kernel_num = 100
kernel_sizes = [1, 2, 3, 4]


class Model(nn.Module):
    def __init__(self, num_aspects=10, dropout=0.5):
        super(Model, self).__init__()
        gen_emb = torch.load("word_embedding/general_embeddings.ts")
        self.gen_embedding = nn.Embedding.from_pretrained(embeddings=gen_emb)
        domain_emb = torch.load("word_embedding/domain_embeddings.ts")
        self.domain_embedding = nn.Embedding.from_pretrained(embeddings=domain_emb)

        self.conv11 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv12 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)

        self.conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(256, 256, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)


        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)
        self.dropout = nn.Dropout(dropout)


        # self.linear_ae = torch.nn.Linear(256, num_classes)

        self.textCNNs = nn.ModuleList([TextCNN() for _ in range(num_aspects)])

    def forward(self, x, x_len):
        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        pack_x_emb = pack_padded_sequence(x_emb, x_len, batch_first=True)
        x_emb, x_len = pad_packed_sequence(pack_x_emb, batch_first=True)

        x_emb = self.dropout(x_emb).transpose(1, 2)

        x_conv = F.relu(torch.cat((self.conv11(x_emb), self.conv12(x_emb)), dim=1))
        x_conv = self.bn1(x_conv)
        x_conv = self.dropout(x_conv)


        x_conv = F.relu(self.conv2(x_conv))
        x_conv = self.bn2(x_conv)
        x_conv = self.dropout(x_conv)

        x_conv = F.relu(self.conv3(x_conv))
        x_conv = self.bn3(x_conv)
        x_conv = self.dropout(x_conv)

        x_conv = F.relu(self.conv4(x_conv))

        # x_emb = self.linear_ae(x_conv)

        x_emb = x_conv.transpose(1, 2)
        # x_emb = x_emb.unsqueeze(1)

        results = []
        for textCNN in self.textCNNs:
            result = textCNN(x_emb)
            results.append(result.unsqueeze(1))

        return torch.cat(results, dim=1)


class TextCNN(nn.Module):
    def __init__(self, class_num=4, channel_size=1, dropout_ratio=0.5):
        """
        :param vocabulary_size: 字典大小
        :param embedding_dim: 词语维度
        :param class_num: 分类种类
        :param kernel_num: 卷积核数量
        :param kernel_sizes: 卷积核大小
        :param channel_size:
        :param dropout_ratio:
        """
        super(TextCNN, self).__init__()

        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_channels=channel_size,
                      out_channels=kernel_num,
                      kernel_size=(kernel_size, 256)),
            nn.BatchNorm2d(kernel_num),
            nn.ReLU(inplace=True),

            # nn.Conv2d(in_channels=kernel_num,
            #           out_channels=kernel_num,
            #           kernel_size=(kernel_size, 1)),
            # nn.BatchNorm2d(kernel_num),
            # nn.ReLU(inplace=True),
        )
            for kernel_size in kernel_sizes])

        self.fc = nn.Sequential(
            nn.Linear(kernel_num * len(kernel_sizes), 300),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(300, 300),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(300, 300),
            nn.BatchNorm1d(num_features=300),
            nn.Dropout(dropout_ratio),
            nn.ReLU(inplace=True),

            nn.Linear(300, class_num),
            nn.LogSoftmax(dim=1),
            nn.BatchNorm1d(num_features=class_num),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = [conv(x).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    gen_embeddings = nn.Embedding.from_pretrained(embeddings=torch.load("../word_embedding/general_embeddings.ts"))
    domain_embeddings = nn.Embedding.from_pretrained(embeddings=torch.load("../word_embedding/domain_embeddings.ts"))

    d = domain_embeddings(torch.tensor([3476]).long())
    g = gen_embeddings(torch.tensor([3476]).long())
    print()
