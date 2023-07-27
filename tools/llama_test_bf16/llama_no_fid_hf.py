from transformers import AutoTokenizer, LlamaForCausalLM
import os
import torch
import json


seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

checkpoint_path ="/opt/amazon/raw_model/meta/llama-13b/"
tokenizer_path = "/opt/amazon/raw_model/meta/llama-13b/"
max_length=50
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model_cpu = LlamaForCausalLM.from_pretrained(checkpoint_path)

prompt = "Hey, are you conscious? Can you talk to me?"
device = "cuda:0"

prompts = [
    "What is the capital of France?",
    "Explain the concept of gravity.",
    "Who wrote the play 'Romeo and Juliet'?",
    "What are the primary colors?",
    "How do plants perform photosynthesis?",
    "What is the boiling point of water?",
    "Tell me about the life of Albert Einstein.",
    "Explain the process of cellular mitosis.",
    "Who painted the 'Mona Lisa'?",
    "What are the main components of a computer?",
    "Describe the structure of a cell.",
    "What are the major continents of the world?",
    "Explain the principles of supply and demand.",
    "Who is the current president of the United States?",
    "What are the benefits of regular exercise?",
    "Describe the water cycle.",
    "What are the different states of matter?",
    "Explain the concept of renewable energy.",
    "Who discovered the theory of relativity?",
    "What are the major organs of the human body?",
    "Tell me about the history of the Roman Empire.",
    "Explain the process of digestion.",
    "Who is the author of 'To Kill a Mockingbird'?",
    "What are the characteristics of mammals?",
    "Describe the major phases of the moon.",
    "What is the significance of the Declaration of Independence?",
    "Explain the process of photosynthesis in plants.",
    "Who wrote the play 'Hamlet'?",
    "What are the primary functions of the nervous system?",
    "Describe the major events of the American Civil War."
]

def generate_from_prompt(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    input_ids=input_ids.to(device)

    # Generate
    generate_ids = model.generate(input_ids, max_length=max_length)
    decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded_output

def generate_from_prompts(model, prompts):
    results = []
    for prompt in prompts:
        print("\n\nprompt:", prompt)
        output = generate_from_prompt(model, prompt)
        print("response:", output, "\n\n")
        results.append({"prompt":prompt, "output": output})
    return results

for precision in [torch.float32, torch.bfloat16, torch.float16]:
    print("-----------------------------------------------------------------------------")
    print("-----------------------------------------------------------------------------")
    print("precision is:", precision)
    model = model_cpu.to(device)
    model = model.to(precision)
    results = generate_from_prompts(model, prompts)
    file_path = f"hf_13b_{str(precision)}.json"
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print("file saved into", file_path)

