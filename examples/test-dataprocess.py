# SPDX-License-Identifier: Apache-2.0

"""
This file showcases how someone can use the data-processing script to
take a dataset from `messages` format into raw input_ids and labels
"""

# Standard
import argparse

# First Party
from instructlab.training.data_process import process_data

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--data-output-path", type=str, required=True)
parser.add_argument("--model-path", type=str, required=False)
parser.add_argument("--legacy", action="store_true", default=False)
parser.add_argument("--chat-tmpl-path", type=str, required=False)
parser.add_argument("--max-seq-len", type=int, required=False, default=2048)

predefined_models = {
    "llama-3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "granite-3.1": "ibm-granite/granite-3.1-8b-instruct",
    "phi-4": "microsoft/phi-4",
    "granite-7b": "ibm-granite/granite-7b-instruct",
}

args = parser.parse_args()

# whether or not we should use one of the pre-defined models
if predefined_model := predefined_models.get(args.model_path):
    model_path = predefined_model
else:
    model_path = args.model_path


process_data(
    data_path=args.data_path,
    data_output_path=args.data_output_path,
    model_path=model_path,
    chat_tmpl_path=args.chat_tmpl_path,
    max_seq_len=args.max_seq_len,
    num_cpu_procs=1,
)
