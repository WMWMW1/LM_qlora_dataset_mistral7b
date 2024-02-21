from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, pipeline
from trl import PPOConfig, AutoModelForCausalLMWithValueHead, PPOTrainer

# 加载和准备数据集
dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"], return_tensors="pt").squeeze(0)
    return sample

# 加载模型和分词器
config = PPOConfig(model_name="gpt2", learning_rate=1.41e-5)
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 使用pipeline作为奖励模型
reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")

# 准备数据集
dataset = dataset.map(tokenize, batched=False)
dataset.set_format(type='torch', columns=['input_ids'])

# 创建自定义DataLoader
standard_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# 初始化PPOTrainer
ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    tokenizer=tokenizer,
)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}
# 训练循环
epochs = 3
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    for batch in standard_dataloader:
        query_tensors = batch["input_ids"]
        print(query_tensors)
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
