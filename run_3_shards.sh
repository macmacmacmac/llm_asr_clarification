#!/bin/bash

# sbatch -o train_shard0.out -e train_shard0.out generate_gold_labels.sh --output-name 32b --shard-id 0 --num-shards 3
# sbatch -o train_shard1.out -e train_shard1.out generate_gold_labels.sh --output-name 32b --shard-id 1 --num-shards 3
# sbatch -o train_shard2.out -e train_shard2.out generate_gold_labels.sh --output-name 32b --shard-id 2 --num-shards 3
# sbatch -o val_shard.out -e val_shard.out generate_gold_labels.sh --split validation --output-name 32b --shard-id 0 --num-shards 1


sbatch -o mask_train0.out -e mask_train0.out generate_gold_labels.sh --split train --output-name 32b_mask_v2 --shard-id 0 --num-shards 4
sbatch -o mask_train1.out -e mask_train1.out generate_gold_labels.sh --split train --output-name 32b_mask_v2 --shard-id 1 --num-shards 4
sbatch -o mask_train2.out -e mask_train2.out generate_gold_labels.sh --split train --output-name 32b_mask_v2 --shard-id 2 --num-shards 4
sbatch -o mask_train3.out -e mask_train3.out generate_gold_labels.sh --split train --output-name 32b_mask_v2 --shard-id 3 --num-shards 4
sbatch -o mask_val.out    -e mask_val.out    generate_gold_labels.sh --split validation --output-name 32b_mask_v2
