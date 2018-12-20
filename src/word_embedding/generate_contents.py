import os
from pyltp import Segmentor

from pandas._libs import json

from const import LTP_DATA_DIR, DATA_SET_DIR

test_file_path = "datafountain/test_public.json"
train_file_path = "datafountain/train.json"


def main():
    cws_model_path = os.path.join(LTP_DATA_DIR, "cws.model")  # 分词模型路径，模型名称为`cws.model`
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型

    train_dict = json.loads(open(os.path.join(DATA_SET_DIR, train_file_path), "r").readline())
    test_dict = json.loads(open(os.path.join(DATA_SET_DIR, test_file_path), "r").readline())
    contents = []

    for value in train_dict.values():
        contents.append(" ".join([word for word in segmentor.segment(value["content"])]))

    for value in test_dict.values():
        contents.append(" ".join([word for word in segmentor.segment(value["content"])]))

    contents_file = open("contents.txt", "w", encoding='utf-8')
    contents_file.write("\n".join(contents))


if __name__ == '__main__':
    main()
