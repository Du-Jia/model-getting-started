from __future__ import annotations

import time
import datetime
from typing import List, Dict, Tuple
import os

import torch
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchtext.data.functional import simple_space_split
import torch.nn.functional as F

from src.data_process import ATISDataProcessor
from src.models.lstm_tagger import BiLSTMSequentialLabelingModel
from src.schema import (Config,
                        InputFeatures,
                        InputExample)
from src.utils import get_time_dif
from src.data_process import DataIterator
from src.metric_report import get_acc


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
    vocab_size += 1
    wv.append([0.]*embed_dim)
    word2id['PAD'] = vocab_size
    vocab_size += 1
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
    config.max_seq_length = 40
    config.hidden_size = 16
    config.num_layers = 4
    config.num_labels = len(config.classes)

    config.output_dir = '.\\data\\ATIS\\save_dict'

    config.num_epochs = 8
    return config


def get_features_from_file(file: str, config: Config) -> List[InputFeatures]:
    print()
    processor: ATISDataProcessor = ATISDataProcessor()
    examples: List[InputExample] = processor.get_examples(file)
    features: List[InputFeatures] = []
    for example in examples:
        feature: InputFeatures = get_single_feature_from_example(example, config)
        features.append(feature)
        # print(feature)
    return features


def get_single_feature_from_example(example: InputExample, config: Config) -> InputFeatures:
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
    max_length = config.max_seq_length
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
    # print(len(ids), len(tokens), len(mask))
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


def train(config: Config, model: BiLSTMSequentialLabelingModel, train_iterator, dev_iterator, test_iterator) -> None:
    """
    A train method for sequential tagger
    Parameters:
        config: Config, it contains some parameters
        model: BiLSTMSequentialLabelingModel, an initialized model
    Returns:
        None
    """
    # record start running time
    start_time = time.time()
    dev_best_loss: float = float('inf')
    # last batch number which improves the result
    last_improve: float = 0
    optimizer = optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    # num of training batches
    batches: int = 0
    # if early stop is True, stop training
    early_stop = False

    for epoch in range(config.epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for (train_x, train_y) in train_iterator.get_batch():
            # set model to train model
            model.train()
            # transform data to current device
            train_x = train_x.to(config.device)
            train_y = train_y.to(config.device)
            try:
                output = model(train_x)
            except:
                print(train_x[-10:])
            model.zero_grad()
            # TODO: Implement a universial loss function to fit different task
            # print(output.shape, train_y.shape)
            # loss = F.multilabel_soft_margin_loss(output, train_y)
            loss_fc = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fc(output.view(-1, config.num_labels), train_y.view(-1))
            print('Batch: {}, train loss: {:0.4}'.format(batches, loss.item()))
            loss.backward()
            optimizer.step()

            if batches % config.save_checkpoints_steps == 0:
                # true = train_y.data.cpu()
                # predict = torch.max(output.data, 1)[1].cpu()
                # train_acc = metrics.accuracy_score(true, predict)
                train_acc = get_acc(output, train_y)
                dev_acc, dev_loss = eval(model, dev_iterator, config, do_test=False)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    t = datetime.datetime.now()
                    state_path = os.path.join(config.output_dir,
                                              f'{t.year}-{t.month}-{t.day}-{t.hour}-lstm-sequential_labeling.ckpt')
                    torch.save(model.state_dict(), state_path)
                    improve = "*"
                    last_improve = batches
                else:
                    improve = ''

                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(batches, loss.item(), train_acc, dev_loss, dev_acc,
                                 get_time_dif(start_time), improve))
                if batches - last_improve > config.require_improve:
                    early_stop = True
                    print('No improving for a long time')
                    break
            batches += 1
            if early_stop:
                break
    # test(model, test_iterator, config, model_file=state_path)


def eval(model, dev_iterator, config, do_test=False):
    model.eval()
    num_batches: int = 0
    loss_sum: float = 0.
    acc_all: float = 0.
    predicted_all: np.ndarray = np.array([], dtype=int)
    labels_all: np.ndarray = np.array([], dtype=int)
    with torch.no_grad():
        for dev_x, dev_y in dev_iterator.get_batch():
            dev_x = dev_x.to(config.device)
            dev_y = dev_y.to(config.device)
            outputs = model(dev_x)
            # Eval Loss
            loss_fc = torch.nn.CrossEntropyLoss(ignore_index=0)
            eval_loss = loss_fc(outputs.view(-1, config.num_labels), dev_y.view(-1))
            loss_sum += eval_loss
            # Metrics
            # labels = dev_y.data.cpu().numpy()  # labels: [batch_size, seq_len, num_labels]
            # predict = torch.argmax(output, dim=-1, keepdim=True).data.cpu().numpy()
            average_acc = get_acc(outputs, dev_y)
            # count = 0
            # for i, pred in enumerate(predict):
            #     count += 1
            #     label = labels[i]
            #     acc = metrics.accuracy_score(pred, label)
            #     average_acc += acc
            # average_acc /= count
            acc_all += average_acc
            # predict = torch.max(output.data, 1)[1].cpu().numpy()  # predict: [batch_size, seq_len, 1]
            # labels_all = np.append(labels_all, labels)
            # predict_all = np.append(predict_all, predict)
            num_batches += 1
    acc_all = acc_all / num_batches
    # acc = metrics.accuracy_score(labels_all, predict_all)
    # if do_test:
    #     report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_sum / num_batches, report, confusion
    return acc_all, loss_sum / num_batches


def test(model, test_iterator, config, model_file):
    pass


if __name__ == '__main__':
    # features = get_features_from_file('data/ATIS/test.json')
    # print(features[0])
    # wv = load_wv()
    # print(wv.vectors.shape, len(wv.vocab))
    print("{}Loading w2v{}".format('*'*20, '*'*20))
    config = create_config()
    print("{}Loaded{}".format('*'*23, '*'*22))
    print(config.pretrained_model.shape, config.pretrained_model)

    features = get_features_from_file('./data/ATIS/train.json', config)
    split: int = int(len(features) * 0.8)
    trainset = features[:split]
    devset = features[split:]
    train_iterator = DataIterator(trainset, config.train_batch_size, config.max_seq_length, config.vocab_size-1)
    dev_iterator = DataIterator(devset, config.eval_batch_size, config.max_seq_length, config.vocab_size)
    testset = get_features_from_file('./data/ATIS/test.json', config)
    test_iterator = DataIterator(testset, config.predict_batch_size, config.max_seq_length, config.vocab_size)

    model = BiLSTMSequentialLabelingModel(config)
    print(model)
    train(config, model, train_iterator, dev_iterator, test_iterator)