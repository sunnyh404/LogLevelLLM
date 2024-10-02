from sys import exit
import os
import time
import sys   

main_path = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

project = sys.argv[1] #expected value "Zookeeper"
file_index = sys.argv[2] #expected value "1"
split_type = sys.argv[3] #expected value "train", "test", "validate"

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 42
num_class = 4 #will be dynamically regenerated based on number of log levels
max_seq_l = 256
lr = 1e-5
num_epochs = 20 #original value = 5
use_cuda = True

BASE_MODEL = sys.argv[4] #expected value "roberta-base"
test_size = sys.argv[5] #expected value "0.9"

ft_project = sys.argv[6] #expected value "train", "test", "validate"

base_model = os.path.basename(BASE_MODEL)

if ft_project == "": #for RQ1
    arguments = "_".join(["finetuned", split_type, project, base_model, str(test_size), file_index])
elif ft_project == "ultimate": #for combined
    arguments = "_".join(["finetuned", split_type, project, base_model, str(test_size), file_index, "u"])
else: #for cross system
    arguments = "_".join(["finetuned", split_type, project, ft_project, base_model, str(test_size), file_index, "x"])

output_file = f"results/{base_model}/{project}/{test_size}/output_{arguments}.json"


import pandas as pd
df = pd.read_json(f"logs/{project}_logs_{split_type}_0.6.json")
df = df[df['index'] == int(file_index)]

import torch
from peft import PeftModel
import transformers
import csv
from tqdm import tqdm
import numpy as np   
import datasets
from datasets import load_dataset, Dataset
import socket
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
    GenerationConfig
)

from transformers import LlamaForCausalLM, GenerationConfig ##removed LlamaTokenizer

if ft_project == "":
    LORA_WEIGHTS = f"saved_model/{base_model}/{project}_{test_size}"
elif ft_project == "ultimate":  # for combined
    LORA_WEIGHTS = f"saved_model/{base_model}/{project}_ultimate"
else:
    LORA_WEIGHTS = f"saved_model/{base_model}/{ft_project}_{test_size}"


if str(test_size) == "0.5":
    if ft_project == "":
        LORA_WEIGHTS = f"saved_model/{base_model}/{project}_0.5"
    else:
        LORA_WEIGHTS = f"saved_model/{base_model}/{ft_project}_5"

llm = base_model.split("-")[0]
if llm == "CodeLlama":
    max_length = 100
    cutoff_len = 2048
    print("using CodeLlama tokenizer")
    from transformers import CodeLlamaTokenizer
    tokenizer = CodeLlamaTokenizer.from_pretrained(BASE_MODEL)
    
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference
    
elif llm == "Llama":
    max_length = 100
    cutoff_len = 2048
    print("using Llama2 tokenizer")
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

###Preparing Llama

print(f"BASE_MODEL: {BASE_MODEL}")
print(f"LORA_WEIGHTS: {LORA_WEIGHTS}")

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if device == "cuda":
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    if str(test_size) in ("0.5", "0.6", "10", "20", "30"):
        
        model = LlamaForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        model = PeftModel.from_pretrained(
            model, LORA_WEIGHTS, torch_dtype=torch.float16, force_download=True
        )
        
    elif str(test_size) in ("0", "5"):
        
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
    generation_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",    # finds GPU
    )

    
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    if str(test_size) in ("0.5", "0.6", "10", "20", "30"):
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    if str(test_size) in ("0.5", "0.6", "10", "20", "30"):
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )
    
# if device != "cpu":
#     model.half()
model.eval()
if torch.__version__ >= "2":
    model = torch.compile(model)



def inference(text):

    sequences = generation_pipe(
        text, 
        max_new_tokens = max_length, 
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # do_sample=False,
        do_sample=False)

    return sequences[0]["generated_text"]

def generate_prompt(instruction, input=None):
    
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Input:
{input}
### Response: """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{instruction}
### Response: """


###
def evaluate(
    instruction,
    input=None,
    temperature=0,
    top_p=1,
    top_k=50,
    num_beams=2,
    max_new_tokens=max_length,
    add_eos_token=True, 
    **kwargs,
):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=1,
        **kwargs,             
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.strip()

def generate_response(template, name):
    return template.format(name)

df_results = pd.DataFrame(columns=["file_index", "prediction"])

shot5 = ""

if str(test_size) == "5":
    df_5shot = pd.read_json(f"logs/{project}_instructions_logs_train_5.json")
    
    for  index, row in df_5shot.iterrows():
        shot5 = shot5 + '### Input: The source code is """' + row['code'] + '""", and the log message is ' + row['constant'] + '. \n\n### Response:The log level is ' + row['log_level'] + '. \n'

for row_index, row in tqdm(df.iterrows(), total=len(df)):

    instruction = f"Between debug, warn, error, trace, and info, choose a suitable log level for the source code provided. {shot5}"
    template = 'The source code is """' + row['code'] + '""", and the log message is ' + row['constant'] + '.'

    PROMPT = generate_prompt(instruction, input=template)
    
    prediction=inference(PROMPT).strip()

    row = [file_index, prediction]
    df_results.loc[row_index] = row
        
df_results.to_json(output_file, orient="records", indent=2)
