import os

from pyltp import Segmentor
import pandas as pd
from pandas._libs import json

import numpy as np

from const import LTP_DATA_DIR, DATA_SET_DIR, ASPECT_DICT

train_file_path = "datafountain/train.csv"


def main():
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    train_file = os.path.join(DATA_SET_DIR, train_file_path)
    words_dict_json = open("words.json", "r")
    words_dict = json.loads(words_dict_json.readline())

    lines = pd.read_csv(train_file,
                        header=0,
                        dtype={"content_id": np.str, "content": np.str,
                               "subject": np.str, "sentiment_value": np.int, "sentiment_word": np.str
                               }, )

    lines_dict = {}
    for index in lines.index:
        line = lines.loc[index]
        line_dict = lines_dict.get(line["content_id"], {
            "sentiment_values": [0 for _ in range(10)],
            "sentiment_word": ["" for _ in range(10)]
        })

        line_dict["subject"] = line["subject"]
        line_dict["content"] = line["content"][1 if line["content"][0] == "\"" else 0:
                                               -1 if line["content"][-1] == "\"" else len(line["content"])]
        line_dict["sentiment_values"][ASPECT_DICT[line["subject"]]] = line["sentiment_value"]
        line_dict["sentiment_word"][ASPECT_DICT[line["subject"]]] = line["sentiment_word"]

        line_dict["content_indexes"] = " ".join(
            [str(words_dict[word]["index"]) for word in segmentor.segment(line_dict["content"])])

        lines_dict[line["content_id"]] = line_dict

    line_dict_json = json.dumps(lines_dict)

    json_file = open("train.json", "w")
    json_file.write(line_dict_json)
    json_file.flush()
    json_file.close()


if __name__ == '__main__':
    main()
