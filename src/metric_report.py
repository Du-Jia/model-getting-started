from sklearn import metrics
import torch


def get_metric_report(output: torch.LongTensor, target: torch.LongTensor) -> float:
    print(output.shape, target.shape)


def get_acc(y_dist: torch.LongTensor, y_true: torch.LongTensor) -> float:
    y_true = y_true.data.cpu().numpy()  # labels: [batch_size, seq_len, num_labels]
    predict = torch.argmax(y_dist, dim=-1, keepdim=True).data.cpu().numpy()
    macro_acc = 0
    count = 0
    for i, pred in enumerate(predict):
        count += 1
        label = y_true[i]
        acc = metrics.accuracy_score(pred, label)
        macro_acc += acc
    macro_acc /= count
    return macro_acc


if __name__ == '__main__':
    outputs = torch.randint(low=0, high=100, size=(10, 10, 10))
    labels = torch.randint(low=0, high=10, size=(10, 10, 1))
    print(get_acc(outputs, labels))