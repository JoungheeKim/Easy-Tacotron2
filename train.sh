export CUDA_VISIBLE_DEVICES=1

python train.py \
    --output_directory=output_ver1 \
    --log_directory=log \
    --n_gpus=1 \
    --training_files=filelists/kss_train.txt \
    --validation_files=filelists/kss_valid.txt \
    --epochs=500