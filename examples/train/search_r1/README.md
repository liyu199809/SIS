# Search-R1 Port Implementation for Verl-Tool

This repository contains a complete port of the [Search-R1](https://github.com/PeterGriffinJin/Search-R1) framework to the Verl-Tool ecosystem, enabling large language models to learn search-enhanced question answering through reinforcement learning.

## ⏩ Quick start

You may use the below command to fire up a quick training. Detailed commands for each step and their explanations are contained in the below sections. 

We assume you:
1. Will be executing the code at the root directory of verl-tool repo.
2. Will store both the training data and retriever index files at `./data/search_r1`.
3. Already have the conda environment: `verl-tool-env` properly configured.

(Note, `faiss-gpu` seems only able to install with conda, so if you encounter error using pip to install it, please use conda to install it.)

### Step 1: Download the retriever index and prepare the retrieval server

```bash
# download the index
save_path=./data/search_r1/retriever_index
python ./verl/examples/sglang_multiturn/search_r1_like/local_dense_retriever/download.py --save_path $save_path

# Prepare index
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

### Step 2: Start the retrieval server
- create a separate environment for the retrieval server, e.g. `search-retriever`, and activate it.
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
uv pip install transformers datasets fastapi numpy torch uvicorn
```
- then run the retrieval server with the following command:

```bash
# activate sglang-retriever
file_path=./data/search_r1/retriever_index
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/wiki-18.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2
python ./verl_tool/servers/tools/utils/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --faiss_gpu &
```
need to wait for 5 mins

### Step 3: prepare the training dataset
```bash
python examples/data_preprocess/search_r1.py --local_dir ./data/search_r1/training_data --prefix_type search_r1
```

### Step 4: perform model training
```bash
# GRPO + SIS
bash ./examples/train/search_r1/train_8B_sis.sh
```

Other variants in this directory: `train_8B_dapo.sh` (DAPO + SIS), `train_8B_gspo.sh` (GSPO + SIS), and `train_7b.sh` (baseline GRPO without SIS, from upstream verl-tool). SIS is toggled by `actor_rollout_ref.actor.use_clip_less` in each script.

For model format conversation and tensorboard visualization, refer to the following sections.

