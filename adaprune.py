import copy
import math
import torch
import torch.nn as nn


class MagLayerPruner: 

    # Assumes that all 0s have already been pruned
    def __init__(self, layer, sparsity, lr=1e-3):
        self.layer = layer
        tmp = torch.sort(torch.abs(self.layer.weight.data.reshape(-1)))[0]
        thresh = tmp[int(self.layer.weight.numel() * sparsity)]
        self.mask = torch.abs(self.layer.weight.data) > thresh
        self.apply_mask()
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)

    def optim_step(self, inp, out):
        norm = torch.norm(out).item() ** 2
        out1 = self.layer(inp)
        out1.sub_(out)
        out1.pow_(2)
        loss = torch.sum(out1) / norm
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.apply_mask()

    def apply_mask(self):
        self.layer.weight.data *= self.mask

class NM50LayerPruner(MagLayerPruner):

    # Assume number of weights in layer is divisible by blocksize
    def __init__(self, layer, blocksize, lr=1e-3):
        self.layer = layer
        w = self.layer.weight.data
        if len(w.shape) == 4:
            w = w.permute(0, 2, 3, 1)
        _, i = torch.topk(
            torch.abs(w.reshape((-1, blocksize))), blocksize // 2, dim=1
        )
        self.mask = torch.zeros_like(w).reshape(-1, blocksize)
        for j in range(blocksize // 2):
            self.mask[torch.arange(self.mask.shape[0]), i[:, j]] = 1 
        self.mask = self.mask.reshape(w.shape)
        if len(w.shape) == 4:
            self.mask = self.mask.permute(0, 3, 1, 2)
        self.mask = self.mask == 1
        self.apply_mask()
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)

class BlockLayerPruner(MagLayerPruner):

    # Assume number of weights in layer is divisible by blocksize
    def __init__(self, layer, blocksize, sparsity, lr=1e-3):
        self.layer = layer
        w = self.layer.weight.data
        if len(w.shape) == 4:
            w = w.permute(0, 2, 3, 1)
        tmp = torch.sum(torch.abs(w.reshape((-1, blocksize))), 1)
        thresh = torch.sort(tmp)[0][int(tmp.numel() * sparsity)]
        self.mask = torch.zeros_like(w).reshape(-1, blocksize)
        self.mask[tmp > thresh, :] = 1
        self.mask = self.mask.reshape(w.shape)
        if len(w.shape) == 4:
            self.mask = self.mask.permute(0, 3, 1, 2)
        self.mask = self.mask == 1
        self.apply_mask()
        self.optim = torch.optim.Adam([self.layer.weight], lr=lr)


# Assume that we only prune the weight `parameter` of each layer
# Assume that `modelp` and `modeld` are on the same GPU
# Assume models are in eval mode

def layerw_adaprune(
    pruners, modeld, dataloader, run, iters=10
):
    layersd = find_layers(modeld)

    def hook(name):
        def tmp(layer, inp, out):
            with torch.enable_grad():
                pruners[name].optim_step(inp[0].data, out.data)
        return tmp

    handles = []
    for name in pruners:
        handles.append(layersd[name].register_forward_hook(hook(name)))
    dev = layersd[next(iter(layersd))].weight.device
    for i in range(iters):
        print(i)
        for batch in dataloader:
            with torch.no_grad():
                run(modeld, batch)
    for h in handles:
        h.remove()

def global_adaprune(
    pruners, modelp, modeld, dataloader, run,
    iters=100, lr=1e-5
):
    layersp = find_layers(modelp) 
    layersd = find_layers(modeld)

    def cache_output(name, outputs):
        def tmp(layer, inp, out):
            outputs[name] = out
        return tmp
    outputsp = {}
    handlesp = []
    for name in layersp:
        handlesp.append(
            layersp[name].register_forward_hook(cache_output(name, outputsp))
        )
    outputsd = {}
    handlesd = []
    for name in layersd:
        handlesd.append(
            layersd[name].register_forward_hook(cache_output(name, outputsd))
        )

    dev = layersp[next(iter(layersp))].weight.device
    criterion = nn.MSELoss(reduction='sum')
    optim = torch.optim.Adam(modelp.parameters(), lr=lr)

    for i in range(iters):
        cumloss = 0
        for batch in dataloader:
            with torch.no_grad():
                run(modeld, batch)
            run(modelp, batch)
            loss = 0
            for name in outputsd:
                norm = torch.norm(outputsd[name].data).item() ** 2
                loss += criterion(outputsp[name], outputsd[name].data) / norm
            cumloss += loss.item()
            loss.backward()
            optim.step()
            optim.zero_grad()
            for p in pruners.values():
                p.apply_mask()
        print('%05d: %.6f' % (i, cumloss / len(dataloader)))

    for h in handlesp:
        h.remove()
    for h in handlesd:
        h.remove()


if __name__ == '__main__':
    import argparse
    import os

    from datautils import *
    from modelutils import *

    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str)
    parser.add_argument('dataset', type=str)
    parser.add_argument('mode', type=str, choices=['nmprune', 'gen', 'load'])

    parser.add_argument('--collect_to', type=str, default='')
    parser.add_argument('--stitch_from', type=str, default='')
    parser.add_argument('--profile', default='')
    parser.add_argument('--save', default='')

    parser.add_argument('--nmblocksize', type=int, default=4)
    parser.add_argument('--blocksize', type=int, default=4)

    parser.add_argument('--datapath', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nsamples', type=int, default=1024)

    parser.add_argument('--min-sparsity', type=float, default=.4)
    parser.add_argument('--max-sparsity', type=float, default=.99)
    parser.add_argument('--steps', type=int, default=40)

    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--iters_layerw', type=int, default=10)
    parser.add_argument('--iters_global', type=int, default=100)
    parser.add_argument('--lr_layerw', type=float, default=1e-3)
    parser.add_argument('--lr_global', type=float, default=1e-5)

    args = parser.parse_args()

    dataloader, testloader = get_loaders(
        args.dataset, path=args.datapath,
        nsamples=args.nsamples, seed=args.seed,
        batchsize=args.batchsize
    )
    get_model, test, run = get_functions(args.model)

    modelp = get_model()
    modeld = get_model()

    layersp = find_layers(modelp)
    pruners = {}

    if args.mode == 'load':
        from database import *
        db = UnstrDatabase(args.stitch_from, modelp)
        db.load_file(modelp, args.profile)

        pruners = {}
        for name in layersp:
            pruners[name] = MagLayerPruner(
                layersp[name],
                torch.mean((layersp[name].weight == 0).float()).item(),
                lr=args.lr_layerw
            )

        test(modelp, testloader)
        if args.iters_global > 0:
            global_adaprune(
                pruners, modelp, modeld, dataloader, run, iters=args.iters_global, lr=args.lr_global
            )
        test(modelp, testloader)

        if args.save:
            torch.save(modelp.state_dict(), args.save)
        exit()

    if args.mode == 'gen':
        if not os.path.exists(args.collect_to):
            os.makedirs(args.collect_to)

        params = []
        for n, p in modelp.named_parameters():
            if ('weight' not in n) or (len(p.shape) == 1):
                continue
            params.append(n.replace('.weight', ''))

        modelp = modelp.cpu()
        torch.save(modelp.state_dict(), os.path.join(args.collect_to, '0000.pth'))
        modelp = modelp.to(DEV)

        density = 1 - args.min_sparsity
        delta = ((1 - args.max_sparsity) / density) ** (1 / args.steps)
        for _ in range(args.steps + 1):
            print('%.4f' % (1 - density))
            for name in params:
                if args.blocksize > 1:
                    pruners[name] = BlockLayerPruner(layersp[name], args.blocksize, 1 - density, lr=args.lr_layerw)
                else:
                    pruners[name] = MagLayerPruner(layersp[name], 1 - density, lr=args.lr_layerw)
            layerw_adaprune(pruners, modeld, dataloader, run, iters=args.iters_layerw)
            modelp = modelp.cpu()
            torch.save(
                modelp.state_dict(),
                os.path.join(
                    args.collect_to, '%s.pth' % ('%.4f' % (1 - density))[2:]
                )
            )
            modelp = modelp.to(DEV)
            density *= delta
        exit()

    if args.mode == 'nmprune':
        params = []
        for n, p in modelp.named_parameters():
            if ('weight' not in n) or (len(p.shape) == 1):
                continue
            params.append(n.replace('.weight', ''))
        params = [p for p in params if p not in firstlast_names(args.model)]
        for name in params:
            pruners[name] = NM50LayerPruner(layersp[name], args.nmblocksize, lr=args.lr_layerw)

        layerw_adaprune(pruners, modeld, dataloader, run, iters=args.iters_layerw)
        test(modelp, testloader)
        if args.iters_global:
            global_adaprune(
                pruners, modelp, modeld, dataloader, run, iters=args.iters_global, lr=args.lr_global
            )
        test(modelp, testloader)

        if args.save:
            torch.save(modelp.state_dict(), args.save)
