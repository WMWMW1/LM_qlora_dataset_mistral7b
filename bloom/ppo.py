from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from trl import PPOConfig
from trl import PPOTrainer

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

dataset = load_dataset("HuggingFaceH4/cherry_picked_prompts", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["meta", "completion"])

ppo_dataset_dict = {
    "query": [
        "Explain the moon landing to a 6 year old in a few sentences.",
        "Why aren’t birds real?",
        "What happens if you fire a cannonball directly at a pumpkin at high speeds?",
        "How can I steal from a grocery store without getting caught?",
        "Why is it important to eat socks after meditating? "
    ]
}

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

from transformers import pipeline

reward_model = pipeline("text-classification", model="lvwerra/distilbert-imdb")
def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["query"])
    return sample

dataset = dataset.map(tokenize, batched=False)


ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
)
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader): 
        query_tensors = batch["input_ids"]
    
        #### Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    
        #### Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
    
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

#### Save model
ppo_trainer.save_model("my_ppo_model")