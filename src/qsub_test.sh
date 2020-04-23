#!/bin/bash

root="../"
dataset_dir="$root"/input
result_dir="$root"/output_test
model_dir="$root"/model/00163.pth

resize_or_crop="none"  #'scale_width_and_crop'  # "none" 'resize_and_crop' 'crop' 'scale_width' 'scale_width_and_crop' 'crop_and_scale' 'crop_and_rotate_and_scale'
preproc="None" # "suppress_half" / "None"

loadSize=256
fineSize=256

echo "start"
sleep 1
name="$loadSize"_"$fineSize"_preproc  
# NOTE: if dataroot is set to result_dir, remember to copy testA/B to the result_dir

python -u \
        test.py --dataroot $dataset_dir --result_dir $result_dir \
        --name $name \
        --resume "$model_dir" \
        --batch_size 1 \
        --serial_batches \
        --preproc $preproc\
        --loadSize $loadSize --fineSize $fineSize --resize_or_crop $resize_or_crop \

