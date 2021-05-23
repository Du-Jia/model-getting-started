from __future__ import annotations
from typing import List, Dict, Tuple
import os

import torch
import torch.nn as nn
import torch.optim as optimizer
from torchtext.data.functional import simple_space_split

from src.data_process import ATISDataProcessor
from src.models.lstm_tagger import BiLSTMSequentialLabelingModel
from src.schema import (Config,
                        InputFeatures,
                        InputExample)
from src.utils import get_time_dif


def load_wv(file: str=r'D:\Data\NLP\pretrained-model\GIoVe\glove.6B.100d.txt') -> Tuple[
    List[List[float]], int, Dict[str, int], Dict[int, str]]:
    wv: List[List] = []
    word2id: Dict[str, int] = {}
    id2word: Dict[int, str] = {}
    vocab_size = 0
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split()
            word = line[0]
            vec = [float(num) for num in line[1:]]
            wv.append(vec)
            word2id[word] = vocab_size
            id2word[vocab_size] = word
            vocab_size += 1
    embed_dim = len(wv[0])

    word2id['UNK'] = vocab_size
    id2word[vocab_size] = 'UNK'
    wv.append([0.]*embed_dim)
    word2id['PAD'] = vocab_size
    id2word[vocab_size] = 'PAD'
    wv.append([0.]*embed_dim)

    return torch.FloatTensor(wv), embed_dim, vocab_size, word2id, id2word


def load_classes(file: str='data\\ATIS\\classes.txt') -> Dict[str, int]:
    classes: Dict[str, int] = {}
    id: int = 0
    with open(file, 'r') as f:
        lines: List[str] = f.readlines()
        for line in lines:
            classes[line.strip()] = id
            id += 1
    return classes


def create_config() -> Config:
    config = Config()
    (config.pretrained_model,
     config.embedding_dim,
     config.vocab_size,
     config.vocab,
     id2word)= load_wv()
    config.classes = load_classes()
    return config


def get_features_from_file(file: str, config: Config) -> List[InputFeatures]:
    processor: ATISDataProcessor = ATISDataProcessor()
    examples: List[InputExample] = processor.get_examples(file)
    features: List[InputFeatures] = []
    for example in examples:
        feature: InputFeatures = get_single_feature_from_example(example, config)
        features.append(feature)
        print(feature)
    return features


def get_single_feature_from_example(example: InputExample, config: Config, max_length=40) -> InputFeatures:
    """
    Transfer an example to a feature. text -> ids list; labels -> label_id list;
    Add padding if text is too long.
    Parameters:
        example: InputExample
    Returns:
        feature: InputFeature
    """
    text = example.text_a
    labels = example.label
    tokens = text.strip().split()
    assert len(labels) == len(tokens)

    # to id
    vocab = config.vocab
    classes = config.classes
    ids = [0] * len(tokens)
    for i, token in enumerate(tokens):
        if token in vocab:
            id = vocab[token]
        else:
            id = vocab['UNK']
        ids[i] = id
    # mask
    mask = [1]*max_length
    pad = vocab['PAD']
    if len(tokens) <= max_length:
        for i in range(max_length - len(tokens)):
            mask[-i-1] = 0
            ids.append(pad)
            labels.append('O')
    else:
        ids = ids[:max_length]
        labels = labels[:max_length]
    print(len(ids), len(tokens), len(mask))
    assert len(ids) == len(mask)
    # label
    label_ids = []
    for label in labels:
        label_id = classes[label]
        label_ids.append(label_id)
    feature = InputFeatures(
        input_ids=ids,
        attention_mask=mask,
        segment_ids=None,
        label_id=label_ids,
    )
    return feature





def train(config: Config, model: BiLSTMSequentialLabelingModel) -> None:
    """
    A train method for sequential tagger
    Parameters:
        config: Config, it contains some parameters
        model: BiLSTMSequentialLabelingModel, an initialized model
    Returns:
        None
    """
    pass


if __name__ == '__main__':
    # features = get_features_from_file('data/ATIS/test.json')
    # print(features[0])
    # wv = load_wv()
    # print(wv.vectors.shape, len(wv.vocab))
    config = create_config()
    print(config.pretrained_model.shape, config.pretrained_model)
    features = get_features_from_file('./data/ATIS/test.json', config)