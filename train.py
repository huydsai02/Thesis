import os
import pandas as pd
import numpy as np
import random
import json
import re

import datasets
from datasets import ClassLabel
from datasets import load_dataset, load_metric, Dataset

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers import Trainer
from transformers import TrainingArguments
from transformers import Wav2Vec2Processor, Wav2Vec2Config
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from w2v_customized_nvlb import Wav2Vec2ForCTC
from wav2vec2_model import Wav2Vec2SmallNonStreamingForCTC
import torch
import torchaudio
import warnings, gc, string, glob
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description = "Distillation_trainer")
parser.add_argument('--train_path', type=str, default="", help='The path of the training data')
parser.add_argument('--initial_model', type=str, default="", help='Path of the initial_model')
parser.add_argument('--streaming',  action='store_true',  help='Turn on streaming mode')
parser.add_argument('--batch_size', type=int,   default=8,     help='Batch size')
args = parser.parse_args()



warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TEACHER_MODEL_PATH = "nguyenvulebinh/wav2vec2-large-vi-vlsp2020"
teacher_processor = Wav2Vec2Processor.from_pretrained(TEACHER_MODEL_PATH, cache_dir=CACHE_DIR)
teacher_model = Wav2Vec2ForCTC.from_pretrained(TEACHER_MODEL_PATH, cache_dir=CACHE_DIR)
teacher_model = teacher_model.eval()
teacher_model = teacher_model.to("cuda")


df_train = pd.read_csv(args.train_path)
df_test = pd.read_csv(args.train_path)

train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# processor = teacher_processor
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side="right",
    do_normalize = not args.streaming, return_attention_mask=False
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=teacher_processor.tokenizer
)

print("VOCAB SIZE:", len(processor.tokenizer))


@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        speech_raw = [torchaudio.load(feature["path"])[0][0].numpy() for feature in features]
        
        batch = self.processor(
            speech_raw,
            sampling_rate=16000,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        teacher_batch = teacher_processor(
            speech_raw,
            sampling_rate=16000,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
    
        with self.processor.as_target_processor():
            labels_batch = self.processor(
                [feature["transcript"] for feature in features],
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
            
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["teacher_input_values"] = teacher_batch['input_values']
        batch["labels"] = labels
#         print(batch)
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

if args.streaming:
    save_checkpoint_folder = os.path.join(os.path.dirname(__file__), "streaming_checkpoint")
else:
    save_checkpoint_folder = os.path.join(os.path.dirname(__file__), "nonstreaming_checkpoint")

os.makedirs(save_checkpoint_folder, exist_ok=True)

info_finetune = {
    "teacher_model": TEACHER_MODEL_PATH,
    "checkpoint_folder": save_checkpoint_folder,
    "need_feature_transform": True,
    "use_one_reshape_layer": False,
    "mse_hidden_states_ratio": 0.2,
    "mse_logits_ratio": 0.3,
    "mem_length": 64,
    "indices_hidden_learn": [i for i in range(0, 25, 2)],
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "FE_f2_conv_dim": 512,
    "type_positional_conv_embedding": "causal",
    "use_streaming_attention_mask": args.streaming,
    "num_frames": 48,
    "num_conv_pos_embeddings": 24
}

with open(os.path.join(info_finetune["checkpoint_folder"], "info_finetune.json"), "w") as outfile: 
    json.dump(info_finetune, outfile, indent=4)


if len(args.initial_model.strip()) == 0:
    checkpoint_folder = os.path.join(CACHE_DIR, "models--nguyenvulebinh--wav2vec2-large-vi-vlsp2020", "*")
    state_dict_path = glob.glob(os.path.join(checkpoint_folder, "pytorch_model.bin"))[0]
    config_path = glob.glob(os.path.join(checkpoint_folder, "config.json"))[0]
else:
    state_dict_path = os.path.join(args.initial_model, "pytorch_model.bin")
    config_path = os.path.join(args.initial_model, "config.json")

state_dict = torch.load(state_dict_path)

with open(config_path, "r") as f:
    student_config = json.load(f)
    
student_config['hidden_size'] = info_finetune["hidden_size"]
student_config["num_hidden_layers"] = info_finetune["num_hidden_layers"]
student_config["intermediate_size"] = info_finetune["intermediate_size"]
student_config["conv_dim"][0] = info_finetune["FE_f2_conv_dim"]
student_config["conv_dim"][1] = info_finetune["FE_f2_conv_dim"]
student_config["num_conv_pos_embeddings"] = info_finetune["num_conv_pos_embeddings"]
student_config["pad_token_id"] = processor.tokenizer.pad_token_id
student_config["vocab_size"] = len(processor.tokenizer)

student_config = Wav2Vec2Config(**student_config)

model = Wav2Vec2SmallNonStreamingForCTC(student_config,
    need_feature_transform=info_finetune["need_feature_transform"], 
    loss_log_path=f'{info_finetune["checkpoint_folder"]}/loss.txt', 
    mse_hidden_states_ratio = info_finetune["mse_hidden_states_ratio"],
    mse_logits_ratio = info_finetune["mse_logits_ratio"],
    indices_hidden_learn = info_finetune["indices_hidden_learn"],
    use_one_reshape_layer = info_finetune["use_one_reshape_layer"],
    mem_length = info_finetune["mem_length"],
    type_positional_conv_embedding=info_finetune["type_positional_conv_embedding"],
    use_streaming_attention_mask=info_finetune["use_streaming_attention_mask"],
    num_frames=info_finetune["num_frames"]
)

for name, param in model.named_parameters():

    if name == "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0":
        if state_dict.get(name, None) is None:
            name = "wav2vec2.encoder.pos_conv_embed.conv.weight_g"
    elif name == "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1":
        if state_dict.get(name, None) is None:
            name = "wav2vec2.encoder.pos_conv_embed.conv.weight_v"

    if state_dict.get(name, None) is not None:
        if param.shape == state_dict[name].shape:
            param.data = state_dict[name]
            print("LOAD FROM STATE DICT LAYER:", name)
        else:
            assert len(param.shape) == len(state_dict[name].shape)
            assert len(param.shape) >= 1 and len(param.shape) <= 3
            param_shape = param.shape
            if len(param_shape) == 1:
                weight = state_dict[name][:param_shape[0]]
            elif len(param_shape) == 2:
                weight = state_dict[name][:param_shape[0], :param_shape[1]]
            elif len(param_shape) == 3:
                weight = state_dict[name][:param_shape[0], :param_shape[1], :param_shape[2]]
            assert param_shape == weight.shape
            param.data = weight.clone()
            print(f"CUT FROM {tuple(state_dict[name].shape)} TO {tuple(param_shape)} LAYER:", name)
    else:
        try:
            if 'weight' in name:
                torch.nn.init.normal_(param.data, mean=0, std=0.02)
    #             torch.nn.init.xavier_uniform_(param.data)
                print("INIT WEIGHT:", name)
            elif 'bias' in name:
                torch.nn.init.constant_(param.data, 0)
                print("INIT WEIGHT:", name)
        except:
            print(f"FAILED TO INIT LAYER:", name)

del state_dict
torch.cuda.empty_cache()
gc.collect()

def teacher_generate_func(input_values):
    with torch.no_grad():
        output = teacher_model(input_values, output_hidden_states=True)
    return output["logits"], output["hidden_states"]
model.set_teacher_generate_func(teacher_generate_func)

if args.streaming:
    epoch = 15
    lr = 1e-4
else:
    epoch = 30
    lr = 5e-4
    model.freeze_feature_extractor()

print("STUDENT MODEL NUM PARAMETER:", model.num_parameters())

training_args = TrainingArguments(
    output_dir=info_finetune["checkpoint_folder"],
    group_by_length=False,
    per_device_train_batch_size=args.batch_size,
    evaluation_strategy="steps",
    num_train_epochs=epoch,
    save_steps=1000,
    eval_steps=1000000000,
    logging_steps=1000,
    dataloader_num_workers=6,
    learning_rate=lr,
    warmup_steps=500,
    save_total_limit=2,
    eval_accumulation_steps=1,
    report_to='tensorboard',
    save_safetensors=False,
    remove_unused_columns=False
)


trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=None,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()
# trainer.train(resume_from_checkpoint=True)