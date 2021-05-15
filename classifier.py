from __future__ import annotations
import datetime
import os
import pickle

import numpy as np
import torch

from src.data_process import THCNewsDataProcessor, DataIterator, THCNewsFeaturesExtractor
from src.models.base_model import SequenceClassificationModel, BiLSTMSequenceClassificationModel
from src.schema import Config
from train import (train, test)


def create_model_config() -> Config:
    config = Config()
    # set data path, contains: train and test file
    root = os.path.join(os.path.abspath('.'), 'data')
    config.data_dir = os.path.join(root, 'THCNews')
    config.language = 'zh'
    # pretrained model path, contains:
    # 1. pretrained model's binary file
    # 2. vocab
    pretrained_path = os.path.join(os.path.join(root, 'pretrained'), config.language)
    config.vocab_file = os.path.join(pretrained_path, 'vocab.pkl')
    config.pretrained_model_name = 'embedding_SougouNews.npz'
    config.pretrained_file = os.path.join(pretrained_path, config.pretrained_model_name)
    # save log with time here
    config.log_dir = os.path.join(config.data_dir, 'log')
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    # save model after training here
    config.output_dir = os.path.join(config.data_dir, 'save_dict')
    if not os.path.exists(config.output_dir):
        os.mkdir(config.output_dir)
    # load pretrained model
    config.pretrained_model = torch.tensor(
        np.load(config.pretrained_file)["embeddings"].astype('float32'))
    config.embedding_dim = config.pretrained_model.size()[1]
    config.hidden_size = 128
    config.num_layers = 2
    config.dropout = 0.5
    config.num_labels = 10
    config.max_seq_length = 40
    config.num_epochs = 8
    config.class_list = []
    with open(os.path.join(config.data_dir, 'class.txt')) as  f:
        lines = f.readlines()
        for line in lines:
            config.class_list.append(line.strip())
    return config


def create_sequence_classification_model(config: Config) -> SequenceClassificationModel:
    model = BiLSTMSequenceClassificationModel(config)
    return model


def get_data_iterator(config: Config) -> (DataIterator, DataIterator, DataIterator):
    data_processor = THCNewsDataProcessor()
    train_file = os.path.join(config.data_dir, 'train.txt')
    train_examples = data_processor.get_examples(train_file)
    dev_file = os.path.join(config.data_dir, 'dev.txt')
    dev_examples = data_processor.get_examples(dev_file)
    test_file = os.path.join(config.data_dir, 'test.txt')
    test_examples = data_processor.get_examples(test_file)
    print(f'Trainset Length: {len(train_examples)}, Example: {train_examples[0]}')
    print(f'Dev Length: {len(dev_examples)}, Example: {dev_examples[0]}')
    print(f'Testset Length: {len(test_examples)}, Example: {test_examples[0]}')

    vocab = pickle.load(open(config.vocab_file, 'rb'))
    train_iterator = THCNewsFeaturesExtractor(vocab, train_examples).get_data_iterator(
        batch_size=config.train_batch_size, max_len=config.max_seq_length, do_test=False)
    dev_iterator = THCNewsFeaturesExtractor(vocab, dev_examples).get_data_iterator(
        batch_size=config.eval_batch_size, max_len=config.max_seq_length, do_test=False)
    test_iterator = THCNewsFeaturesExtractor(vocab, test_examples).get_data_iterator(
        batch_size=config.predict_batch_size, max_len=config.max_seq_length, do_test=True)
    return train_iterator, dev_iterator, test_iterator


def main():
    config = create_model_config()
    # config = config.parse_args(known_only=True)
    # 0. Load vocab
    vocab = pickle.load(open(config.vocab_file, 'rb'))
    config.n_vocab = len(vocab)
    # 1. load data iterator
    train_iterator, dev_iterator, test_iterator = get_data_iterator(config)
    model = create_sequence_classification_model(config)
    print(f'Modle Info: \n{model}')
    model = model.to(config.device)
    train(config, model, train_iterator, dev_iterator, test_iterator)
    test(model, test_iterator, config)


if __name__ == '__main__':
    main()
