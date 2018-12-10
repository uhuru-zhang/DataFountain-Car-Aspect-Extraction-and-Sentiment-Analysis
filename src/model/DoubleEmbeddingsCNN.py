import torch
import torch.nn as nn
import torch.nn.functional as F

class_num = 4
kernel_num = 100
kernel_sizes = [1, 2, 3, 4]


class Model(nn.Module):
    def __init__(self, gen_emb, domain_emb, num_aspects=3, dropout=0.5):
        super(Model, self).__init__()

        self.gen_embedding = nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)

        self.conv11 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 5, padding=2)
        self.conv12 = nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], 128, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = nn.Conv1d(256, 256, 5, padding=2)

        self.textCNNs = nn.ModuleList([TextCNN() for _ in range(num_aspects)])

    def forward(self, x, x_len):
        # x_emb = nn.utils.rnn.pack_padded_sequence(x_emb, x_len, batch_first=True)

        x_emb = torch.cat((self.gen_embedding(x), self.domain_embedding(x)), dim=2)
        x_emb = self.dropout(x_emb).transpose(1, 2)

        x_conv = F.relu(torch.cat((self.conv11(x_emb), self.conv12(x_emb)), dim=1))
        x_conv = self.dropout(x_conv)

        x_conv = F.relu(self.conv2(x_conv))
        x_conv = self.dropout(x_conv)

        x_conv = F.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)

        x_conv = F.relu(self.conv4(x_conv))
        x_conv = x_conv.transpose(1, 2)

        x_emb = self.linear_ae(x_conv)

        x_emb = x_emb.transpose(1, 2)
        x_emb = x_emb.unsqueeze(1)

        results = []
        for textCNN in self.textCNNs:
            result = textCNN(x_emb)
            results.append(result.unsqueeze(2))

        return torch.cat(results, dim=2)


class TextCNN(nn.Module):
    def __init__(self, channel_size=1, dropout_ratio=0.5):
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
            nn.Linear(kernel_num * len(kernel_sizes), 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),

            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),

            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),

            nn.Linear(100, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_ratio),

            nn.Linear(100, class_num),
            nn.BatchNorm1d(num_features=class_num),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)
        x = self.fc(x)

        return x
