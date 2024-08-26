import torch

from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline
)

local_run = True

class TF_Pipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.pipe = pipeline(
            task="text-generation", 
            model=model, 
            tokenizer=tokenizer
        )
    def get_tfpipeline(self):
        return self.pipe
    

base_model_name = "" # specify the local model dir

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    low_cpu_mem_usage=True,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
pipe = TF_Pipeline(model, tokenizer)


prompt = ""

resp = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=2048, 
    temperature=0.4,
    top_p=0.9,
    num_return_sequences=1,
    )

output = resp[0]['generated_text']


