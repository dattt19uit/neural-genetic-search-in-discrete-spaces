VICTIM_NAME=${1}
SEED=${2:-42}

EXP_NAME=${VICTIM_NAME}_${CLASSIFIER}
if [[ "$CLASSIFIER" == "llama" ]]; then
    EXP_NAME=${EXP_NAME}-${LLAMAGUARD}
fi
EXP_NAME=${EXP_NAME}_seed${SEED}

# python main.py \
# --exp_name $EXP_NAME \
# --sim_tolerance 0.3 \
# --victim_model $VICTIM_NAME \
# --lr 1e-4 \
# --reward_sched_horizon 1000 \
# --train_steps 5000 \
# --buffer_size 5000 \
# --seed $SEED \
# --max_len 20 \
# --temp_low 0.7 \
# --temp_high 2.0 \
# --lm_sched_end 1.2 \
# --lm_sched_horizon 2000 \
# --compare c_reward \
# --prioritization c_reward \
# --beta 0.1 \
# --metric cosine \
# --classifier $CLASSIFIER \
# --llama_guard_version $LLAMAGUARD

# python eval.py \
# --ckpt save/${EXP_NAME}/latest \
# --victim_model $VICTIM_NAME \
# --classifier $CLASSIFIER \
# --llama_guard_version $LLAMAGUARD \
# --no_lora

THRESHOLD_TOXIC=0.7
THRESHOLD_LM=100

# python collect_samples.py --exp_name $EXP_NAME --threshold_toxic $THRESHOLD_TOXIC --threshold_lm $THRESHOLD_LM

SFT_CKPT=save/gpt2-sft-position-final/latest
DISTILLATION_EXP_NAME=${EXP_NAME}-distillation-thres${THRESHOLD_TOXIC}_${THRESHOLD_LM}
PROMPT_FILE=offline_dataset/${EXP_NAME}/dataset.json

STEPS=1000
BS=1024

python main.py \
--mode distillation \
--exp_name $DISTILLATION_EXP_NAME \
--lr 1e-4 \
--seed $SEED \
--batch_size $BS \
--train_steps $STEPS \
--grad_acc_steps 8 \
--model_name $SFT_CKPT \
--few_shot_file $PROMPT_FILE \
--save_dir "./save" \
--classifier $CLASSIFIER \
--llama_guard_version $LLAMAGUARD

python eval.py \
--ckpt save/${DISTILLATION_EXP_NAME}/latest \
--victim_model $VICTIM_NAME \
--classifier $CLASSIFIER \
--llama_guard_version $LLAMAGUARD \
--no_lora
