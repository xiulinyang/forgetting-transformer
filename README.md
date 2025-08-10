## Instructions
To load models from HF, please install the dependencies first
```python
!pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
!pip install pytest einops numpy
!pip install torch==2.4.0
!pip install transformers==4.44.0
# No guarantee other commits would work; we may fix this later
!pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
```

Example code for ```forgetting_transformer```
```python
import forgetting_transformer.model  # Needed to register the model classes
import forgetting_transformer.tokenizer  # Needed to register the tokenizer class
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "xiulinyang/forgetting_transformer"

model = AutoModelForCausalLM.from_pretrained(
    repo,
    torch_dtype="auto",
    device_map="auto",        
)
tokenizer = AutoTokenizer.from_pretrained(repo)

print("model_type:", model.config.model_type)
print("vocab_size:", model.config.vocab_size, "tok.vocab_size:", tokenizer.vocab_size)

inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model.eval()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```