import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(epoch, opt, optimizer):
    if epoch in opt.lr_decay_epochs:
        for param_group in optimizer.param_groups:
            if epoch <= 30:
                param_group['lr'] = param_group['lr'] * opt.lr_decay_rate[0]
            else:
                param_group['lr'] = param_group['lr'] * opt.lr_decay_rate[1]

def group_parameters(model):
    pretrain_parameters = model.state_dict()
    # k[7:22] != "deep_net.layer4"
    base_dict = {k: v for k, v in pretrain_parameters.items() if k[7:23] != "radio_net.layer3" and
                 k[7:23] != "radio_net.layer2" and
                 k[7:23] != "radio_net.layer1" and k[7:22] != "deep_net.layer4"}

    new_parameters = []
    for pname, p in model.named_parameters():
        if pname not in list(base_dict.keys()):
            new_parameters.append(p)

    new_parameters_id = list(map(id, new_parameters))
    base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    parameters = {'base_parameters': base_parameters,
                  'new_parameters': new_parameters}
    # print(model.state_dict().keys())
    print(len(parameters["base_parameters"]), len(parameters["new_parameters"]))
    return parameters


def str2list(string):
    if string == "":
        return []
    iterations = string.split(',')
    res = list([])
    for it in iterations:
        res.append(int(it))
    return res


def cal_score(truth_labels, pred_labels):
    # score = f1_score(truth_labels, pred_labels, average='weighted')
    score = f1_score(truth_labels, pred_labels)
    return score


def score_list(truth_labels, pred_labels, save_folder):
    save_path = save_folder[:-1]
    count_num = save_folder[-1]
    file = open(save_path + "result.txt", "a")
    file.write("---------  " + str(count_num) + "  ---------\n")
    file.write(classification_report(truth_labels, pred_labels, digits=4))
    file.write(str(confusion_matrix(truth_labels, pred_labels)))
    file.write("\n")
    file.close()

