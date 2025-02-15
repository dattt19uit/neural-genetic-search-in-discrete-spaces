GPU=${1:-0}


CUDA_VISIBLE_DEVICES=$GPU python main.py \
--mode sft \
--lr 3e-5 \
--train_steps 100 \
--grad_acc_steps 32 \
--batch_size 1024 \
--prompt_file ./prompts/attack_prompt.jsonl \
--few_shot_file ./prompts/sft_dataset.json \
--exp_name gpt2-sft-position-final