from __future__ import annotations

from src.data_process import AgNewsDataProcessor
from src.data_process import ATISDataProcessor

# def test_agnews_data_processor():
#     """test news data processors
#     """
#     processor = AgNewsDataProcessor()
#     assert len(processor.get_labels()) == 4
#
#     examples = processor.get_train_examples('./data/agnews/train.csv')
#     assert len(examples) > 0


def test_atis_data_processor():
    processor = ATISDataProcessor()

    examples = processor.get_train_examples('../data/ATIS/train.json')
    print(len(examples))


if __name__ == '__main__':
    test_atis_data_processor()