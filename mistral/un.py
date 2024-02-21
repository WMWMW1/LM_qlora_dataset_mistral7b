from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from datasets import load_dataset
from tqdm import tqdm

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = 1000,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = 1000,
)




dataset = load_dataset("stingning/ultrachat", split="train[:20000]")
dataset = dataset.train_test_split(test_size=0.1)

def prepare_dialogue(example):
    text = ""
    for idx, msg in enumerate(example["data"]):
        if idx % 2 == 0:
            text += f"<|user|>\n{msg}{tokenizer.eos_token}\n"
        else:
            text += f"<|assistant|>\n{msg}{tokenizer.eos_token}\n"
    example["text"] = text
    return example

dataset = dataset.map(prepare_dialogue, num_proc=4, remove_columns=["id", "data"])

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()