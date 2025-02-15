MODEL_NAME=${1:-llama-3}
SEED=${2:-42}
GPUID=${3:-0}

VICTIM=${MODEL_NAME}
CLASSIFIER=llama
LLAMAGUARD=3

CKPT=${MODEL_NAME}_llama-3_seed42-distillation-thres0.7_100

if [ "$CLASSIFIER" == "harmbench" ]; then
    if [ "$VICTIM" == "llama-3.3" ]; then
        GPUUTIL=0.5  # must be a100l x 4
    elif [ "$VICTIM" == "llama-3" ]; then
        GPUUTIL=0.4  # must be a100l
    else
        GPUUTIL=0.7  # can be l40s
    fi
else
    if [ "$VICTIM" == "llama-3.3" ]; then
        GPUUTIL=0.125  # must be a100l x 2
    else
        GPUUTIL=0.25  # can be l40s
    fi
fi

# NGO decoding
# 1. n_pop: 128
# 2. n_off: 16 or 32 
# 3. mutation_rate: 0.05
# 4. rank_coef: 0.01
# 5. delta: 0.99 or 0.999
# 6. novelty_rank_weight: 0.1 or 0.2
# for N_POP in 128 256; do
#     for N_OFF in 16 32; do
#         for MU in 0.05 0.1; do
#             for DELTA in 0.95 0.99 0.999; do
#                 for NOVRANK in 0.1 0.2; do
#                     echo "Running with n_pop=$N_POP, n_off=$N_OFF, mu=$MU, delta=$DELTA, novelty_rank_weight=$NOVRANK"
#                     python eval_seed_topp.py \
#                         --ckpt save/${CKPT}/latest \
#                         --victim_model $VICTIM \
#                         --classifier $CLASSIFIER \
#                         --llama_guard_version $LLAMAGUARD \
#                         --no_lora \
#                         --seed $SEED \
#                         --gpu_util $GPUUTIL \
#                         --decoding ngo \
#                         --n_pop $N_POP \
#                         --n_off $N_OFF \
#                         --mutation_rate $MU \
#                         --delta $DELTA \
#                         --novelty_rank_weight $NOVRANK \
#                         --rank_coef 0.01;
#                 done
#             done
#         done
#     done
# done

N_POP=128
N_OFF=32
MU=0.1
DELTA=0.99
NOVRANK=0.1
TEMP=1.0
CUDA_VISIBLE_DEVICES=$GPUID python eval_final.py \
    --ckpt save/${CKPT}/latest \
    --victim_model $VICTIM \
    --classifier $CLASSIFIER \
    --llama_guard_version $LLAMAGUARD \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding ngo \
    --temp $TEMP \
    --n_pop $N_POP \
    --n_off $N_OFF \
    --mutation_rate $MU \
    --delta $DELTA \
    --novelty_rank_weight $NOVRANK \
    --rank_coef 0.01 \
