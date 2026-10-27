#!/bin/bash

sbatch -o train_shard0.out -e train_shard0.out generate_gold_labels.sh --output-name 32b --shard-id 0 --num-shards 3
sbatch -o train_shard1.out -e train_shard1.out generate_gold_labels.sh --output-name 32b --shard-id 1 --num-shards 3
sbatch -o train_shard2.out -e train_shard2.out generate_gold_labels.sh --output-name 32b --shard-id 2 --num-shards 3
