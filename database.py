# Helper code for handling a the reconstruction database.


from collections import * 

import torch
import torch.nn as nn

from modelutils import *


class UnstrDatabase:

    def __init__(self, path, model, skip=[]):
        self.db = defaultdict(OrderedDict)
        denselayers = find_layers(model)
        dev = next(iter(denselayers.values())).weight.device
        for f in os.listdir(path):
            sparsity = '0.' + f.split('.')[0]
            sd = torch.load(os.path.join(path, f), map_location=dev)
            for layer in denselayers:
                if layer not in skip:
                    self.db[layer][sparsity] = sd[layer + '.weight']

    def layers(self):
        return list(self.db.keys())

    def load(self, layers, name, config='', sd=None):
        if sd is not None:
            layers[name].weight.data = sd[name + '.weight']
            return
        layers[name].weight.data = self.db[name][config]

    def stitch(self, layers, config):
        for name in config:
            self.load(layers, name, config[name])

    def load_file(self, model, profile):
        config = {}
        with open(profile, 'r') as f:
            for line in f.readlines():
                splits = line.split(' ')
                sparsity = splits[0]
                name = splits[1][:-1]
                config[name] = sparsity
        for name in self.db:
            if name not in config:
                config[name] = '0.0000'
        layers = find_layers(model)
        self.stitch(layers, config)

    def get_errors(self):
        errors = {}
        for name in self.db:
            errors[name] = {}
            for i, s in enumerate(sorted(self.db[name])):
                errors[name][s] = (i / (len(self.db[name])- 1)) ** 2
        return errors 

    def get_params(self, layers):
        tot = 0
        res = {}
        for name in layers:
            res[name] = {}
            tot += layers[name].weight.numel()
            for sparsity in self.db[name]:
                res[name][sparsity] = torch.sum(
                    (self.db[name][sparsity] != 0).float()
                ).item()
        return tot, tot, res

    def get_timings(self, path):
        timings = {}
        with open(path, 'r') as f:
            lines = f.readlines()
            baselinetime = float(lines[1])
            prunabletime = float(lines[3])
            i = 4
            while i < len(lines):
                name = lines[i].strip()
                timings[name] = {}
                i += 1
                while i < len(lines) and ' ' in lines[i]:
                    time, level = lines[i].strip().split(' ')
                    timings[name][level] = float(time)
                    i += 1
        timings = {l: timings[l] for l in self.layers()}
        return baselinetime, prunabletime, timings
