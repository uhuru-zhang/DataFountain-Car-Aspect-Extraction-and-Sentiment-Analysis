import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as D
from tensorboardX import SummaryWriter

from data_set.data_set import DatafountainDataSet
from model.DoubleEmbeddingsCNN import Model

class_num = 3


def get_input_and_target(contents, aspect_level):
    b_content_indexes = [[int(index_str) for index_str in content.split(" ")] for content in contents]

    index_lengths = [(i, len(content_indexes)) for i, content_indexes in enumerate(b_content_indexes)]

    lengths = [length for _, length in sorted(index_lengths, key=lambda a: a[1], reverse=True)]
    indexes = [index for index, _ in sorted(index_lengths, key=lambda a: a[1], reverse=True)]

    content_indexes_padding = [content_indexes + (lengths[0] - len(content_indexes)) * [0] for
                               content_indexes in
                               b_content_indexes]

    data = torch.tensor(content_indexes_padding, dtype=torch.long)
    data = torch.index_select(data, dim=0, index=torch.Tensor(indexes).long())

    target = torch.cat([t.unsqueeze(1) for t in aspect_level], dim=1)
    target = torch.index_select(target, dim=0, index=torch.Tensor(indexes).long())
    # target = target.eq(3)

    return data, torch.tensor(lengths), target.long()


def main():
    writer = SummaryWriter()

    train_loader = D.DataLoader(DatafountainDataSet(train=True),
                                batch_size=256,
                                shuffle=True, num_workers=32)

    test_loader = D.DataLoader(DatafountainDataSet(train=False),
                               batch_size=256,
                               shuffle=True, num_workers=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model().to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adadelta(params=model.parameters())

    globals_index = 0
    for epoch in range(1, 1000000):
        train_loss, train_accuracy = 0, 0
        model.train()
        for batch_index, line in enumerate(train_loader):
            contents, aspect_level = line[1], line[2]
            optimizer.zero_grad()

            data, lengths, target = get_input_and_target(contents, aspect_level)
            target = target.to(device)

            output = model(data.to(device), lengths.to(device))

            loss = F.cross_entropy(input=output.view(output.shape[0] * 10, 4), target=target.view(target.shape[0] * 10))
            loss.backward()
            optimizer.step()

            pred = output.max(2)[1]
            correct = pred.eq(target).sum().item()

            strict_correct_num = 0
            for pred, actual in zip(output.chunk(output.shape[0], dim=0), target.chunk(target.shape[0], dim=0)):
                pred = pred.squeeze()
                pred = pred.max(1)[1]
                actual = actual.squeeze()

                if torch.equal(pred, actual):
                    strict_correct_num += 1

            train_loss, train_accuracy, strict_train_accuracy = loss.item(), correct / len(
                contents) / 10, strict_correct_num / len(contents)
            if batch_index % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f})]\tLoss: {}\t, Accuracy: ({:.2f}%)\t, Strict Accuracy: ({:.2f}%)".format(
                        epoch, batch_index * len(data), len(train_loader.dataset),
                               100. * batch_index / len(train_loader), loss.item(),
                               100. * correct / len(contents) / 10,
                               100. * strict_correct_num / len(contents),

                    ))

            writer.add_scalar("train/accuracy", train_accuracy, global_step=globals_index)
            writer.add_scalar("train/accuracy", strict_train_accuracy, global_step=globals_index)
            writer.add_scalar("train/loss", train_loss, global_step=globals_index)
            globals_index += 1

        print("epoch {} done!".format(epoch))

        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0
            strict_correct_num = 0

            for batch_index, line in enumerate(test_loader):
                contents, aspect_level = line[1], line[2]
                optimizer.zero_grad()

                data, lengths, target = get_input_and_target(contents, aspect_level)
                target = target.to(device)

                output = model(data.to(device), lengths.to(device))

                test_loss += F.cross_entropy(input=output.view(output.shape[0] * 10, 4),
                                             target=target.view(target.shape[0] * 10))

                pred = output.max(2)[1]
                correct += pred.eq(target).sum().item()

                for pred, actual in zip(output.chunk(output.shape[0], dim=0), target.chunk(target.shape[0], dim=0)):
                    pred = pred.squeeze()
                    pred = pred.max(1)[1]
                    actual = actual.squeeze()

                    if torch.equal(pred, actual):
                        strict_correct_num += 1

            test_loss /= len(test_loader)

            print(
                "\nTest set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.2f}%), Test Strict Accuracy: ({:.2f}%), \n"
                    .format(test_loss, correct, len(test_loader.dataset),
                            100. * correct / len(test_loader.dataset) / 10,
                            100. * strict_correct_num / len(test_loader.dataset)))

            writer.add_scalars("accuracy", {"train": train_accuracy, "test": correct / len(test_loader.dataset) / 10},
                               epoch)
            writer.add_scalars("strict accuracy",
                               {"train": train_accuracy, "test": strict_correct_num / len(test_loader.dataset)}, epoch)
            writer.add_scalars("lost", {"train": train_loss, "test": test_loss}, epoch)


if __name__ == '__main__':
    main()
