#!/bin/bash

batch1=256
traj_enc='bow'
K=25
gpu_id=0
length=1
group=1

sobel='' #'--sobel'
prefix='paction_kmeans+gmm_spherical_'
data='/data/ajabri/Penn_Action/frames'

clustering='Kmeans'
clustering2='Kmeans'
reg_cov=0.001

## NEED TO TRY WITH PIC TOO

for length in $length
do
    # gpu_id=length-1
    name=${prefix}_${clustering}_${traj_enc}_T${length}_K${K}_group${group}${sobel} #_regcov${reg_cov}
    echo ${name}

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 dc_main.py $data \
    --workers 20 $sobel --arch resnet18 --pretrained --no-blur --frame-size 256 \
    --batch $batch1 --verbose --exp /tmp/${name} --group ${group} \
    --k ${K} --ep_length -1 --traj_length ${length} --traj_enc ${traj_enc} --epochs 10 \
    --export 0 --dump-html 0 --clustering ${clustering} --reg_covar ${reg_cov} #--lr 0.001 

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