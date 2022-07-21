# SPDY: Accurate Pruning with Speedup Guarantees

This repository contains reference impelementations of all methods introduced in our ICML 2022 paper: [SPDY: Accurate Pruning with Speedup Guarantees](https://arxiv.org/abs/2201.13096).
This includes the DP algorithm for efficiently solving constrained layer-wise compression problems (see `dpsolve()` in `spdy.py`), the reparametrized SPDY search for injecting global information into otherwise purely layer-wise problems (see `spdy.py`), as well as the enhanced post trainig pruning method global AdaPrune (see `adaprune.py`).

## Usage:

The code depends on `torch` and `torchvision`.
The following block shows sample commands for the various features of the repository.
See `--help` of `adaprune.py` and `spdy.py` for additional options.

```
# Path to ImageNet
export DATAPATH = path/to/imagenet

# 2:4 AdaPrune + global AdaPrune
python adaprune.py rn18 imagenet nmprune --datapath ${DATAPATH}
# 4:8 AdaPrune + global AdaPrune
python adaprune.py rn18 imagenet nmprune --nmblocksize 8 --datapath ${DATAPATH}

# Generate unstr database
python adaprune.py rn18 imagenet gen --collect_to rn18_unstr --datapath ${DATAPATH}
# Generate 4block database
python adaprune.py rn18 imagenet gen --blocksize 4 --collect_to rn18_4block --datapath ${DATAPATH}

# Run SPDY search to find 2x speedup profile
python spdy.py rn18 imagenet rn18_unstr timings/rn18_unstr.txt 2 rn18_unstr_200x.txt --datapath ${DATAPATH}

# Load and evaluate profile & run global AdaPrune
python adaprune.py rn18 imagenet load --stitch_from rn18_unstr --profile rn18_unstr_200x.txt --datapath ${DATAPATH}
```

Currently, the repository supports several torchvision-ResNet variants. However, all the core features are implemented so that they can also easily be applied to other models by providing a few corresponding small wrapper functions, see `modelutils.py` and `datautils.py` for their ResNet implementations.

## Cite:

If you found this work useful, please consider citing:

```
@article{frantar-spdy,
  title={{SPDY}: Accurate Pruning with Speedup Guarantees}, 
  author={Elias Frantar and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2201.13096}
}
```
