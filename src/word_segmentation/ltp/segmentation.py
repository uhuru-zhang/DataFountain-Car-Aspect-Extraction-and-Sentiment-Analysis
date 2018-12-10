import os

from pyltp import Segmentor
import pandas as pd
from pandas._libs import json

import numpy as np

from const import LTP_DATA_DIR, DATA_SET_DIR

train_file_path = "datafountain/test_public.csv"


# validationset_file_path = "validationset/sentiment_analysis_validationset.csv"


def main():
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    train_file = os.path.join(DATA_SET_DIR, train_file_path)
    # validationset_file = os.path.join(DATA_SET_DIR, validationset_file_path)

    #  content_id, content, subject, sentiment_value, sentiment_word
    lines = pd.read_csv(train_file,
                        header=0,
                        dtype={"content_id": np.str, "content": np.str,
                               # "subject": np.str, "sentiment_value": np.int, "sentiment_word": np.str
                               },
                        index_col="content_id", )

    lines = lines["content"]
    json_file = open("words.json", "r")
    word_dict = json.loads(json_file.readline())
    for line in lines:
        words = segmentor.segment(line[1 if line[0] == "\"" else 0:
                                       -1 if line[-1] == "\"" else len(line)])
        regist_words(word_dict, words)

    word_dict_json = json.dumps(word_dict)

    json_file = open("words.json", "w")
    json_file.write(word_dict_json)
    json_file.flush()
    json_file.close()


def regist_words(word_dict, words):
    for word in words:
        if word in word_dict:
            word_dict[word]["num"] += 1
        else:
            word_dict[word] = {
                "num": 1,
                "index": word_dict[""]["index"]
            }
            word_dict[""]["index"] += 1


if __name__ == '__main__':
    main()
