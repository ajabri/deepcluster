#!/bin/bash

gpu_id=$1
data=$2
prefix="${3}_paction_aff_14x14_"

#data='/data/ajabri/Penn_Action/frames'

echo ${gpu_id}
batch1=128
traj_enc='bow'
K=50
length=2
group=1
N=40000
sobel='' #'--sobel'
frame_skip=4

clustering='Kmeans'
clustering2='Kmeans'
pretrained="--pretrained"
# pretrained="" #"--pretrained"

reg_cov=0.001
workers=20

flow=1

## NEED TO TRY WITH PIC TOO

for length in $length
do
    # gpu_id=length-1
    name=${prefix}_${clustering}_${traj_enc}_T${length}_K${K}${pretrained}_N${N}_fskip${frame_skip}_flow${flow} #_regcov${reg_cov}
    echo ${name}
    
    CUDA_VISIBLE_DEVICES=${gpu_id} python3 dc_main.py $data --N ${N}  \
    --workers ${workers} $sobel --arch resnet18 ${pretrained} --no-blur --frame-size 256 \
    --batch $batch1 --verbose --exp /tmp/${name} --group ${group} \
    --k ${K} --ep_length -1 --traj_length ${length} --traj_enc ${traj_enc} --epochs 30 \
    --export 1 --dump-html 1 --clustering ${clustering} --reg_covar ${reg_cov} \
    --optical-flow ${flow} --frame-skip ${frame_skip}
    
    # CUDA_VISIBLE_DEVICES=${gpu_id} python3 export_clusters.py $data \
    # --workers 20 --group ${group} $sobel --clustering ${clustering2} \
    # --batch $batch2 --verbose --resume /tmp/${name}/checkpoint.pth.tar \
    # --k ${K} --traj_length ${length} --traj_enc ${traj_enc} --reg_covar ${reg_cov}
    # exit
    
done


# ./run.sh bow 50 0 &
# ./run.sh bow 100 1 &
# ./run.sh temp_conv 50 2 &
# ./run.sh temp_conv 100 3 &
