import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"
hf_token = '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'
pipeline = transformers.pipeline(
    "text-generation", 
    model=model_id, 
    model_kwargs={"torch_dtype": torch.bfloat16}, 
    device_map="auto",
    token=hf_token,
    max_new_tokens=50  # Adjust this value as needed
)

pipeline("Hey how are you doing today?")
