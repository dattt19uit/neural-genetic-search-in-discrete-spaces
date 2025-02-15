SOURCE_VICTIM=${1}

EXP_NAME=${SOURCE_VICTIM}-gfn

python main.py \
--exp_name $EXP_NAME \
--sim_tolerance 0.3 \
--victim_name $SOURCE_VICTIM \
--lr 1e-4 \
--reward_sched_horizon 1000 \
--train_steps 5000 \
--buffer_size 5000 \
--seed 42 \
--max_len 20 \
--temp_low 0.7 \
--temp_high 2.0 \
--lm_sched_end 1.2 \
--lm_sched_horizon 2000 \
--compare c_reward \
--prioritization c_reward \
--beta 0.1 \
--metric cosine \

python eval.py \
--ckpt save/${EXP_NAME}/latest \
--victim_name $SOURCE_VICTIM \
--no_lora

python collect_samples.py --exp_name $EXP_NAME

DISTILLATION_EXP_NAME=${EXP_NAME}-distillation

python main.py \
--mode distillation \
--exp_name $DISTILLATION_EXP_NAME \
--lr 1e-4 \
--seed $SEED \
--batch_size 1024 \
--train_steps 1000 \
--grad_acc_steps 8 \
--model_name save/gpt2-sft-position-final/latest \
--few_shot_file offline_dataset/${EXP_NAME}/dataset.json \
--save_dir "./save" \

python eval.py \
--ckpt save/${DISTILLATION_EXP_NAME}/latest \
--victim_name $SOURCE_VICTIM \
--no_lora
