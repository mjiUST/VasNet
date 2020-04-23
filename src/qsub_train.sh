#!/bin/bash

root="../"
dataset_dir="$root"/input
result_dir="$root"/output_train

resize_or_crop="crop_and_rotate_and_scale"  #'scale_width_and_crop'  # "none" 'resize_and_crop' 'crop' 'scale_width' 'scale_width_and_crop' 'crop_and_scale' 'crop_and_rotate_and_scale'
loadSize=256
fineSize=256

echo "start"
name=DANN-"$loadSize"_"$fineSize"_"$resize_or_crop"  # percep_styleB
python -u train-new_dataloader.py --dataroot $dataset_dir --result_dir $result_dir --display_dir $result_dir \
        --name $name \
        --display_id -1 \
        --batch_size 2 --lambdaB 0.1 --lr 0.0002 --model_save_freq 1 --n_ep 300 \
        --loadSize $loadSize --fineSize $fineSize --resize_or_crop $resize_or_crop
