
# d16, 256x256
# torchrun \
#   --nnodes=1 \
#   --node_rank=0 \
#   --nproc_per_node=4 \
#   --master_addr=127.0.0.1 \
#   --master_port=7777 \
#   train.py \
#   --data_path=/data2/jiangpf/data/imagenet-1k \
#   --depth=16 \
#   --bs=128 \
#   --ep=200 \
#   --fp16=1 \
#   --alng=1e-3 \
#   --wpe=0.1

# d20, 256x256
# torchrun \
#   --nnodes=1 \
#   --node_rank=0 \
#   --nproc_per_node=4 \
#   --master_addr=127.0.0.1 \
#   --master_port=7777 \
#   train.py \
#   --data_path=/data2/jiangpf/data/imagenet-1k \
#   --depth=20 \
#   --bs=768 \
#   --ep=250 \
#   --fp16=1 \
#   --alng=1e-3 \
#   --wpe=0.1

# d24, 256x256
# torchrun \
#   --nnodes=1 \
#   --node_rank=0 \
#   --nproc_per_node=4 \
#   --master_addr=127.0.0.1 \
#   --master_port=7777 \
#   train.py \
#   --data_path=/data2/jiangpf/data/imagenet-1k \
#   --depth=24 \
#   --bs=48 \
#   --ep=350 \
#   --tblr=8e-5 \
#   --fp16=1 \
#   --alng=1e-4 \
#   --wpe=0.01 \
#   --use_implicit=True

# d30, 256x256
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=127.0.0.1 \
  --master_port=7777 \
  train.py \
  --data_path=/data2/jiangpf/data/imagenet-1k \
  --depth=30 \
  --bs=96 \
  --ep=350 \
  --tblr=8e-5 \
  --fp16=1 \
  --alng=1e-5 \
  --wpe=0.01 \
  --twde=0.08 \
  --use_implicit=True
