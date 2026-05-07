
# d30, 256x256
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=7777 \
  train.py \
  --data_path=/path/to/imagenet-1k \
  --depth=30 \
  --bs=512 \
  --ep=550 \
  --tblr=8e-5 \
  --fp16=1 \
  --alng=1e-5 \
  --wpe=0.01 \
  --twde=0.08 \
  --use_implicit=True \
  --p_depth=5
