CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --cfg models/yolov5s.yaml \
    --weights yolov5s.pt \
    --data coco.yaml \
    --hyp data/hyps/hyp.scratch.yaml \
    --sparseml-recipe yolov5s_150x_spdy.yaml \
    --batch-size 128 \
    --patience 0
