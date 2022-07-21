import os
import sys

import torch
import torch.nn as nn


DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def test(model, dataloader):
    train = model.training
    model.eval()
    print('Evaluating ...')
    dev = next(iter(model.parameters())).device
    preds = []
    ys = []
    for x, y in dataloader:
        preds.append(torch.argmax(model(x.to(dev)), 1))
        ys.append(y.to(dev))
    acc = torch.mean((torch.cat(preds) == torch.cat(ys)).float()).item()
    acc *= 100
    print('%.2f' % acc)
    if train:
        model.train()

def get_test(name):
    return test


def run(model, batch, loss=False, retmoved=False):
    dev = next(iter(model.parameters())).device
    if retmoved:
        return (batch[0].to(dev), batch[1].to(dev))
    out = model(batch[0].to(dev))
    if loss:
        return nn.functional.cross_entropy(out, batch[1].to(dev)).item()
    return out

def get_run(model):
    return run


from torchvision.models import *

get_models = {
    'rn18': lambda: resnet18(pretrained=True),
    'rn34': lambda: resnet34(pretrained=True),
    'rn50': lambda: resnet50(pretrained=True),
    'rn101': lambda: resnet101(pretrained=True),
    '2rn50': lambda: wide_resnet50_2(pretrained=True)
}

def get_model(model):
    model = get_models[model]()
    model = model.to(DEV)
    model.eval()
    return model


def get_functions(model):
    return lambda: get_model(model), get_test(model), get_run(model)


def firstlast_names(model):
    if 'rn' in model:
        return ['conv1', 'fc']
