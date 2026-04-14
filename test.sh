export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

LOG=${save_dir}"res.log"
shot=(1 2 4)
depth=(9)
n_ctx=(12)
t_n_ctx=(4)

for s in "${!shot[@]}";do
    for i in "${!depth[@]}";do
        for j in "${!n_ctx[@]}";do
        ## train on the mvtec dataset
            timestamp=$(date +%Y%m%d_%H%M%S)
            base_dir=${timestamp}_${shot[s]}_${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}
            save_dir=./checkpoints/${base_dir}/
            python main.py \
            --dataset mvtec3d \
            --data_path /home/js/js/Projects/_datasets/mvtec3d \
            --save_path ./results/${base_dir}/ \
            --checkpoint_path ${save_dir}epoch_20.pth \
            --features_list 6 12 18 24 \
            --image_size 224 \
            --depth ${depth[i]} \
            --n_ctx ${n_ctx[j]} \
            --t_n_ctx ${t_n_ctx[0]} \
            --shot ${shot[s]}
        wait
        done
    done
done

