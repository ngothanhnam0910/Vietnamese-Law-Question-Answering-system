# model=$PWD/{args.output_dir} # path to model
model=/home/ivirse/ivirse_all_data/namnt/llm/merge_model # path to model
num_shard=1             # number of shards
max_input_length=1024   # max input length
max_total_tokens=2048   # max total tokens
 
docker run -d --name tgi --gpus all -ti -p 8080:80 \
  -e MODEL_ID=/workspace \
  -e NUM_SHARD=$num_shard \
  -e MAX_INPUT_LENGTH=$max_input_length \
  -e MAX_TOTAL_TOKENS=$max_total_tokens \
  -v $model:/workspace \
  ghcr.io/huggingface/text-generation-inference:latest
