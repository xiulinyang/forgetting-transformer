from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_dir = "/workspace/forgetting-transformer/fox_saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Hello! Teach me something fun about transformers."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(out[0], skip_special_tokens=True))