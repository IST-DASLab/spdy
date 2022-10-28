# SPDY: Accurate Pruning with Speedup Guarantees

This repository contains reference implementations of all methods introduced in our ICML 2022 paper: [SPDY: Accurate Pruning with Speedup Guarantees](https://arxiv.org/abs/2201.13096).
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

## YOLOv5

In order to work with YOLOv5 models you need to install all packages in `yolov5/requirements.txt` and then create a COCO calibration data folder as follows.

```
cp yolov5/make_calib.py COCO_ROOT
cd COCO_ROOT
python make_calib.py
```

After this YOLO pruning can be performed in similar fashion as described above for ResNet18.

For gradual pruning, we used SparseML v0.9; our integration is contained in the `yolov5` folder. For more extensive work we would however recommend to use the [official integration](https://github.com/neuralmagic/sparseml/tree/main/integrations/ultralytics-yolov5) which supports the newest versions of YOLOv5 and SparseML. Sample SparseML pruning recipes with all our hyper-parameters (e.g. `yolov5s_150x_spdy.yaml`) and launch scripts (e.g. `yolov5s_150x_spdy.sh`) can be found in our `yolov5` folder as well.

## Cite:

If you found this work useful, please consider citing:

```
@inproceedings{frantar-spdy,
  title={{SPDY}: Accurate Pruning with Speedup Guarantees}, 
  author={Elias Frantar and Dan Alistarh},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```
