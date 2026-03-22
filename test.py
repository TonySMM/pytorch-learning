"""
DPO微调脚本 (Direct Preference Optimization)
- 框架: TRL 0.29.0 + PEFT LoRA
- 数据: data/dpo_train (需要 prompt, chosen, rejected 字段)
- 输出: models/qwen2.5-0.5b-dpo
"""
import torch
import warnings
warnings.filterwarnings("ignore")

from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir = "models/qwen2.5-0.5b-dpo"

    print("=== DPO 训练开始 ===")
    print(f"基座模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_from_disk("data/dpo_train")
    eval_dataset  = load_from_disk("data/dpo_test")

    # 只保留必要字段
    keep = ["prompt", "chosen", "rejected"]
    train_dataset = train_dataset.select_columns([c for c in keep if c in train_dataset.column_names])
    eval_dataset  = eval_dataset.select_columns([c for c in keep if c in eval_dataset.column_names])

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        eval_strategy="epoch",
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("开始 DPO 训练...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== DPO 训练完成，模型已保存至 {output_dir} ===")

if __name__ == "__main__":
    main()
