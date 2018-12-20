import os
import torch

from pandas._libs import json

from const import DATA_SET_DIR


def generate_word_dict(file_path):
    words = []
    embeddings = []
    embeddings_file = open(os.path.join(DATA_SET_DIR, file_path), encoding="utf-8")

    embeddings_file.readline()
    for line in embeddings_file:
        words.append(line.split(" ")[0])
        embeddings.append([float(value) for value in line.strip().split(" ")[1:]])

    return words, {word: index for index, word in enumerate(words)}, embeddings


def generate_word_vector_dict(file_path):
    word_vector_dict = {}
    embeddings_file = open(os.path.join(DATA_SET_DIR, file_path), encoding="utf-8")

    embeddings_file.readline()
    for line in embeddings_file:
        word_vector_dict[line.split(" ")[0]] = [float(value) for value in line.rstrip().split(" ")[1:]]

    return word_vector_dict


def save_general_embeddings():
    general_word_vector_dict = generate_word_vector_dict("pre-word-embedding/sgns.zhihu.bigram-char")
    domain_word_vector_dict = generate_word_vector_dict("datafountain/word_vectors.txt")

    general_word_dict_file = open(os.path.join(DATA_SET_DIR, "datafountain/general_word_dict.json"), "r",
                                  encoding="utf-8")
    general_word_dict = json.loads(general_word_dict_file.readline())
    word_tuple = list(sorted([(index, word) for word, index in general_word_dict.items()], key=lambda x: x[0]))

    i = 0
    general_embeddings = []
    domain_embeddings = []
    for index, word in word_tuple:
        if i != index:
            raise Exception("my aaaaa")
        i += 1

        embeddings = general_word_vector_dict[word] if word in general_word_vector_dict else domain_word_vector_dict[word]
        if len(embeddings) < 300:
            print(word)

        general_embeddings.append(embeddings)
        domain_embeddings.append(domain_word_vector_dict.get(word, [0] * 300))

    x = torch.tensor(general_embeddings)
    torch.save(x, "general_embeddings.ts")
    torch.save(torch.tensor(domain_embeddings), "domain_embeddings.ts")


def main():
    save_general_embeddings()

    # general_words, general_word_dict, general_embeddings = generate_word_dict(
    #     "pre-word-embedding/sgns.zhihu.bigram-char")
    # domain_words, domain_word_dict, domain_embeddings = generate_word_dict("datafountain/word_vectors.txt")
    #
    # un_include = {word: index for word, index in domain_word_dict.items() if word not in general_word_dict}
    #
    # print(len(general_words), len(domain_words), len(un_include))
    #
    # for word, index in un_include.items():
    #     general_words.append(word)
    #     general_embeddings.append(domain_embeddings[index])
    #
    # general_word_dict = {word: index for index, word in enumerate(general_words)}
    #
    # general_word_dict_file = open("general_word_dict.json", "w", encoding="utf-8")
    # general_word_dict_file.write(json.dumps(general_word_dict))
    # general_word_dict_file.flush()
    # general_word_dict_file.close()


if __name__ == '__main__':
    main()
