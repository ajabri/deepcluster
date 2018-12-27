#!/bin/bash

# echo $0
# echo $1
# exit
# for traj_enc in 'bow' 'temp_conv'
# do
#     for K in 50 100
#     do
#         for length in 1 3 5 8
#         do
#             gpu_id=length-1
#             name=plain_${traj_enc}_T${length}_K${K}
#             echo ${name}
#             CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py /data3/ajabri/vizdoom/single_env_hard_plain/0 --workers 20\
#             --batch 256 --verbose --exp /tmp/${name} \
#             --k ${K} --traj_length ${length} --traj_enc ${traj_enc} --epochs 2 && \
#             CUDA_VISIBLE_DEVICES=${gpu_id} python3 export_clusters.py /data3/ajabri/vizdoom/single_env_hard_plain/0 --workers 10 \
#             --batch 128 --verbose --resume /tmp/${name}/checkpoint.pth.tar \
#             --k ${K} --traj_length ${length} --traj_enc ${traj_enc} 

#             exit
#         done
#     done
# done

# for traj_enc in 'bow' 'temp_conv'
# traj_enc=$1
# K=$2
# gpu_id=$3
batch1=256
batch2=256
traj_enc='bow'
K=50
gpu_id=0,1
length=5
group=50
sobel='--sobel'
prefix='test_gmm_'
data='/data3/ajabri/vizdoom/single_env_hard_plain/0'

for length in $length
do
    # gpu_id=length-1
    name=${prefix}_${traj_enc}_T${length}_K${K}_group${group}${sobel}
    echo ${name}

    # CUDA_VISIBLE_DEVICES=${gpu_id} python3 main.py $data \
    # --workers 20 $sobel \
    # --batch $batch1 --verbose --exp /tmp/${name} --group ${group} \
    # --k ${K} --traj_length ${length} --traj_enc ${traj_enc} --epochs 50

    CUDA_VISIBLE_DEVICES=${gpu_id} python3 export_clusters.py $data \
    --workers 20 --group ${group} $sobel \
    --batch $batch2 --verbose --resume /tmp/${name}/checkpoint.pth.tar \
    --k ${K} --traj_length ${length} --traj_enc ${traj_enc} 

    # exit
done


# ./run.sh bow 50 0 &
# ./run.sh bow 100 1 &
# ./run.sh temp_conv 50 2 &
# ./run.sh temp_conv 100 3 &