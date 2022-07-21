# SPDY search & DP algorithm implementation.


import copy
import math
import random

import numpy as np
import torch

from modelutils import *


class SPDY:

    def __init__(
        self,
        target, db, errors, baselinetime, prunabletime, timings,
        get_model, run, dataloader,
        skip_layers=[], dpbuckets=10000
    ):
        self.target = target
        self.db = db
        self.run = run
        self.dpbuckets = dpbuckets

        self.modelp = get_model()
        self.layersp = find_layers(self.modelp)

        self.batches = []
        for batch in dataloader:
            self.batches.append(run(self.modelp, batch, retmoved=True))

        self.layers = list(db.layers())
        self.layers = [l for l in self.layers if l not in skip_layers]
        self.sparsities = [list(errors[self.layers[l]].keys()) for l in range(len(self.layers))]
        self.costs = [
            [errors[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]
        self.timings = [
            [timings[self.layers[l]][s] for s in self.sparsities[l]] for l in range(len(self.layers))
        ]

        self.baselinetime = baselinetime
        self.prunabletime = prunabletime
        targettime = self.baselinetime / self.target - (self.baselinetime - self.prunabletime)
        best = sum(min(c) for c in self.timings)
        if self.prunabletime < self.baselinetime:
            print('Max target:', self.baselinetime / (best + self.baselinetime - self.prunabletime))
        self.bucketsize = targettime / self.dpbuckets

        for row in self.timings:
            for i in range(len(row)):
                row[i] = int(round(row[i] / self.bucketsize))

        print('Loss/Base:', self.get_loss(self.modelp))

    def dp(self, costs):
        DP = np.full((len(costs), self.dpbuckets + 1), float('inf'))
        PD = np.full((len(costs), self.dpbuckets + 1), -1)

        for sparsity in range(len(costs[0])):
            if costs[0][sparsity] < DP[0][self.timings[0][sparsity]]:
                DP[0][self.timings[0][sparsity]] = costs[0][sparsity]
                PD[0][self.timings[0][sparsity]] = sparsity
        for layer in range(1, len(DP)):
            for sparsity in range(len(costs[layer])):
                timing = self.timings[layer][sparsity]
                score = costs[layer][sparsity]
                if timing == 0:
                    tmp = DP[layer - 1] + score
                    better = tmp < DP[layer]
                    if np.sum(better):
                        DP[layer][better] = tmp[better]
                        PD[layer][better] = sparsity
                    continue
                if timing > self.dpbuckets:
                    continue
                tmp = DP[layer - 1][:-timing] + score
                better = tmp < DP[layer][timing:]
                if np.sum(better):
                    DP[layer][timing:][better] = tmp[better]
                    PD[layer][timing:][better] = sparsity

        score = np.min(DP[-1, :])
        timing = np.argmin(DP[-1, :])
        
        solution = []
        for layer in range(len(DP) - 1, -1, -1):
            solution.append(PD[layer][timing])
            timing -= self.timings[layer][solution[-1]]
        solution.reverse()
        return solution

    def gen_costs(self, coefs):
        return [
            [self.costs[i][j] * coefs[i] for j in range(len(self.costs[i]))] \
            for i in range(len(self.costs))
        ]

    def stitch_model(self, solution):
        model = copy.deepcopy(self.modelp)
        layers = find_layers(model)
        config = {
            self.layers[i]: self.sparsities[i][solution[i]] for i in range(len(self.layers))
        }
        self.db.stitch(layers, config)
        return model

    @torch.no_grad()
    def get_loss(self, model):
        loss = 0
        for batch in self.batches:
            loss += self.run(model, batch, loss=True)
        return loss / len(self.batches) 

    def get_score(self, coefs):
        costs = self.gen_costs(coefs)
        solution = self.dp(costs)
        model = self.stitch_model(solution)
        return self.get_loss(model)

    def save_profile(self, coefs, filename=''):
        solution = self.dp(self.gen_costs(coefs))
        if filename:
            with open(filename, 'w') as f:
                for i in range(len(solution)):
                    f.write('%s %s\n' % (self.sparsities[i][solution[i]], self.layers[i]))
        else:
            for i in range(len(solution)):
                print('%s %s' % (self.sparsities[i][solution[i]], self.layers[i]))

    def score(self, filename):
        tmp = []
        with open(filename, 'r') as f:
            solution = []
            for i, l in enumerate(f.readlines()):
                splits = l.split(' ')
                sparsity = splits[0]
                tmp.append(float(sparsity))
                name = splits[1][:-1]
                j = self.sparsities[i].index(sparsity)
                solution.append(j)

        print('Speedup:', self.baselinetime / (
            self.baselinetime - self.prunabletime + \
            sum(t[s] for s, t in zip(solution, self.timings)) * self.bucketsize
        ))

        model = self.stitch_model(solution)
        print('Loss/Pruned:', self.get_loss(model))
        return model

    def dpsolve(self, save=''):
        coefs = np.ones(len(self.layers))
        print('Loss/Pruned:', self.get_score(coefs))
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)

    def search(
        self, save='', randinits=100, maxnoimp=100, layerperc=.1
    ):
        evals = 0
        print('Finding init ...')
        coefs = None
        score = float('inf')
        for _ in range(randinits):
            coefs1 = np.random.uniform(0, 1, size=len(self.layers))
            score1 = self.get_score(coefs1)
            evals += 1
            print('%04d  %.4f %.4f' % (evals, score, score1))
            if score1 < score:
                score = score1
                coefs = coefs1
        print('Running local search ...')
        for resamplings in range(round(layerperc * len(self.layers)), 0, -1):
            print('Trying %d resamplings ...' % resamplings)
            improved = True
            while improved: 
                improved = False
                for _ in range(maxnoimp):
                    coefs1 = coefs.copy()
                    for _ in range(resamplings):
                        coefs1[random.randint(0, len(self.layers) - 1)] = np.random.uniform(0, 1)
                    score1 = self.get_score(coefs1)
                    evals += 1
                    print('%04d  %.4f %.4f' % (evals, score, score1))
                    if score1 < score:
                        score = score1
                        coefs = coefs1
                        improved = True
                        break
        self.save_profile(coefs)
        if save:
            self.save_profile(coefs, save)


if __name__ == '__main__':
    import argparse

    from database import *
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str, choices=get_models,
        help='Model to work with.'
    )
    parser.add_argument(
        'dataset', type=str, choices=DEFAULT_PATHS,
        help='Dataset to use.'
    )
    parser.add_argument(
        'database', type=str,
        help='Database location.'
    )
    parser.add_argument(
        'timings', type=str,
        help='Timings file.'
    )
    parser.add_argument(
        'target', type=float,
        help='Target speedup.'
    )
    parser.add_argument(
        'profile', type=str,
        help='Where to save the resulting profile.'
    )

    parser.add_argument(
        '--datapath', type=str, default='',
        help='Path to dataset.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed to use for calibration set selection.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=1024,
        help='Number of samples in the calibration dataset.'
    )

    args = parser.parse_args()

    get_model, test, run = get_functions(args.model)
    dataloader, testloader = get_loaders(args.dataset, noaug=True, nsamples=args.nsamples)

    model = get_model()
    db = UnstrDatabase(args.database, model, skip=firstlast_names(args.model))
    errors = db.get_errors()
    baselinetime, prunabletime, timings = db.get_timings(args.timings)

    spdy = SPDY(
        args.target, db, errors, baselinetime, prunabletime, timings,
        get_model, run, dataloader
    )

    PROFILE_FILE = args.profile
    spdy.search(PROFILE_FILE)
