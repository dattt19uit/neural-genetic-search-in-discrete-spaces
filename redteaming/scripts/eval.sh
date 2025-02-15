SOURCE_VICTIM=${1}
SEED=${2:-42}
TARGET_VICTIM="${3:-$SOURCE_VICTIM}"

CKPT=${SOURCE_VICTIM}-gfn-distillation

if [ "$TARGET_VICTIM" == "llama-3.3" ]; then
    GPUUTIL=0.125  # must be a100l x 2
else
    GPUUTIL=0.25  # can be l40s
fi

# NGS
N_POP=128
N_OFF=32
MU=0.1
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding ngs \
    --n_pop $N_POP \
    --n_off $N_OFF \
    --mutation_rate $MU;

# Sampling (Best-of-N)
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding sampling;

# Tempered
TEMP=0.8
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding tempered \
    --temp $TEMP;

TEMP=0.5
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding tempered \
    --temp $TEMP;

# Top-k
TOPK=10
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding topk \
    --top_k $TOPK;

TOPK=5
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding topk \
    --top_k $TOPK;

# Top-p
TOPP=0.8
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding topp \
    --top_p $TOPP;

TOPP=0.5
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding topp \
    --top_p $TOPP;

# Beam search
BEAMSIZE=4
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding beamsearch \
    --beam_size $BEAMSIZE;

BEAMSIZE=8
python eval.py \
    --ckpt save/${CKPT}/latest \
    --victim_name $TARGET_VICTIM \
    --no_lora \
    --seed $SEED \
    --gpu_util $GPUUTIL \
    --decoding beamsearch \
    --beam_size $BEAMSIZE;
