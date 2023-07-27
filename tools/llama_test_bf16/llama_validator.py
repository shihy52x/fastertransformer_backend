#!/usr/bin/env python3

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import google.protobuf.json_format
import json
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from argparse import ArgumentParser
from collections.abc import Mapping
from tritonclient.utils import np_to_triton_dtype
from transformers import AutoTokenizer, LlamaForCausalLM
import torch

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
max_length=50



def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--params")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("-m", "--model", default="llama")
    parser.add_argument("-i", "--input", help="Directory with input requests.")
    parser.add_argument("-tn", "--tokenizer_name", default="llama", help="Tokenizer implementation.")
    parser.add_argument("-t", "--tokenizer", help="Path to tokenizer configuration")
    args = parser.parse_args()

    return args


def generate_request(input_ids):
    input_lengths = input_ids.shape[1]
    request = [
        {
            'name': 'input_ids',
            'data': np.array(input_ids, dtype='uint32')
        }, {
            'name': 'input_lengths',
            'data': np.array([[input_lengths] for i in range(0, 1)], dtype='uint32')
        }, {
            'name': 'request_output_len',
            'data': np.array([[max_length] for i in range(0, 1)], dtype='uint32')
        }
    ]
    return request


def generate_parameters(args):
    DEFAULT_CONFIG = {
        'protocol': 'grpc',
        'url': None,
        'model_name': args.model,
        'verbose': False,
        'stream_api': False,
    }
    params = {'config': DEFAULT_CONFIG, 'request': []}

    args_params = json.loads(args.params) if args.params else {}
    deep_update(params, args_params)  

    if params['config']['url'] is None:
        if params['config']['protocol'] == 'grpc':
            params['config']['url'] = 'localhost:8001'
        else:
            params['config']['url'] = 'localhost:8000'

    return params['config']


def prepare_tensor(client, name, input):
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def generate_output_ids(config, request):
    is_http = config['protocol'] == 'http'
    client_type = httpclient if is_http else grpcclient
    kwargs = {"verbose": config["verbose"]}
    if is_http:
        kwargs["concurrency"] = 1
    with client_type.InferenceServerClient(config['url'], **kwargs) as cl:
            payload = [prepare_tensor(client_type, field['name'], field['data']) for field in request]
            result = cl.infer(config['model_name'], payload)
            for output in result.get_response().outputs:
                if output.name == 'output_ids':
                    output_ids = result.as_numpy(output.name)
    return output_ids



def generate_from_prompt(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']
    input_ids=input_ids.to("cuda:0")

    # Generate
    generate_ids = model.generate(input_ids, max_length=max_length)
    decoded_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded_output

def get_hf_result(checkpoint_path, tokenizer, output_path):
    model_cpu = LlamaForCausalLM.from_pretrained(checkpoint_path)
    results=[]
    for precision in [torch.float32, torch.bfloat16, torch.float16]:
        model = model_cpu.to("cuda:0")
        model = model.to(precision)
        for prompt in prompts:
            output = generate_from_prompt(model, prompt)
            results.append({"prompt":prompt, "output": output})
            print("response:", output, "\n\n")

        file_path = f"hf_13b_{str(precision)}.json"
        with open(file_path, "w") as json_file:
            json.dump(results, json_file, indent=4)


if __name__ == "__main__":
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

    args = parse_args()
    tokenizer_path= "/opt/amazon/raw_model/meta/llama-13b/"
    # hf_checkpoint_path = "/opt/amazon/raw_model/meta/l/7blama-7b/"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    np.set_printoptions(linewidth=np.inf, threshold=np.inf)
    config = generate_parameters(args)
    results = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt", max_length=50).input_ids
        request = generate_request(input_ids)
        output_ids = generate_output_ids(config,request)
        output = str(tokenizer.decode(output_ids[0][0], skip_special_tokens=True))
        results.append({"prompt":prompt, "output": output})
        print("\n\nprompt:", prompt)
        print("response:", output, "\n\n")
    file_path = "ft_13b_torch.bfloat16.json"
    with open(file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print("file saved into", file_path)

