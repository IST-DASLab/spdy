CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --cfg models/yolov5m.yaml \
    --weights yolov5m.pt \
    --data coco.yaml \
    --hyp data/hyps/hyp.scratch.yaml \
    --sparseml-recipe yolov5m_200x_spdy.yaml \
    --batch-size 128 \
    --patience 0
