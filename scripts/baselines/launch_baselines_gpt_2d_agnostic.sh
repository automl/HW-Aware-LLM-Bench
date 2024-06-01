#!/bin/bash
methods=("RS" "MOREA" "LS" "NSGA2" "LSBO" "RSBO" "MOASHA" "EHVI")
devices=("a100" "rtx2080" "h100" "cpu_xeon_silver" "cpu_xeon_gold") #"a100" "a40" "h100" "rtx2080" "rtx3080" "a6000" "v100" "P100" "cpu_xeon_silver" "cpu_xeon_gold" "cpu_amd_7502" "cpu_amd_7513" "cpu_amd_7452")
metrics=("flops" "params" "float16_memory" "bloat16_memory")
search_spaces=("m" "s" "l")
seeds=(9001 9002 9003 9004 9005) # 9006 9007 9008 9009 9010)
metrics_agnostic=("flops" "params" "float16_memory" "bfloat16_memory")
for seed in "${seeds[@]}"
do 
  for method in "${methods[@]}"
    do
      for search_space in "${search_spaces[@]}"
      do
        for metric in "${metrics_agnostic[@]}"
        do
            exp_name="baseline-${method}-${search_space}-${metric}-${seed}"
            echo Submitting job $exp_name
            sbatch --bosch -J $exp_name scripts/baselines/baselines_gpt_2d_agnostic.sh $search_space $method $metric $seed
        done
      done
    done
done