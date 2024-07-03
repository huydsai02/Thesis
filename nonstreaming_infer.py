# from transformers.file_utils import cached_path, hf_bucket_url
from importlib.machinery import SourceFileLoader
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor)
from wav2vec2_model import Wav2Vec2SmallNonStreamingForCTC
# from IPython.lib.display import Audio
import torchaudio
import torch, os
import pandas as pd 
from jiwer import wer
import numpy as np
from tqdm import tqdm
from pyctcdecode import Alphabet
from pyctcdecode.decoder import BeamSearchDecoderCTC
from pyctcdecode.language_model import LanguageModel
import kenlm, os, time, argparse
from safetensors import safe_open

tqdm.pandas()

parser = argparse.ArgumentParser(description = "Streaming_inference")
parser.add_argument('--checkpoint', type=str, default="", help='Path of the checkpoint')
parser.add_argument('--audio_path', type=str, default="", help='Path of the audio')
parser.add_argument('--language_model_path', type=str, default="", help='Path of the language_model')
parser.add_argument('--syllables_path', type=str, default="", help='Path of the syllables')
args = parser.parse_args()

CHECKPOINT_FOLDER = args.checkpoint

print("CHECKPOINT:", CHECKPOINT_FOLDER)

model = Wav2Vec2SmallNonStreamingForCTC.from_pretrained(
    CHECKPOINT_FOLDER,
    need_feature_transform=True,
    mse_hidden_states_ratio=0, #Remove Reshape layer
    use_streaming_attention_mask=False,
    mem_length=192, num_frames=32,
    type_positional_conv_embedding="causal"
)

print("STUDENT MODEL NUM PARAMETER:", model.num_parameters())

teacher_processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-large-vi-vlsp2020")
# feature_extractor = Wav2Vec2FeatureExtractor(
#     feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side="right",
#     do_normalize=False, return_attention_mask=False
# )
# processor = Wav2Vec2Processor(
#     feature_extractor=feature_extractor, tokenizer=teacher_processor.tokenizer
# )
processor = teacher_processor

model = model.eval().cuda()

def get_decoder_ngram_model(tokenizer):
    vocab_dict = tokenizer.get_vocab()
    sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
    # vocab = [x[1] for x in sort_vocab][:-2]
    vocab = [x[1] for x in sort_vocab]
    vocab_list = vocab
    # specify ctc blank char index, since conventially it is the last entry of the logit matrix
    with open(args.syllables_path, 'r') as f:
        unigram_list = [t.lower() for t in f.read().strip().split("\n")]
    alphabet = Alphabet.build_alphabet(vocab_list)
    kenlm_model = LanguageModel(
        kenlm.Model(args.language_model_path),
        alpha=0.5,
        beta=1.5,
        unigrams=unigram_list,
    )
    decoder = BeamSearchDecoderCTC(alphabet, language_model=kenlm_model)
    return decoder

decoder = get_decoder_ngram_model(processor.tokenizer)

def generate_transcript(path):   
    # Load an example audio (16k)
    audio, sample_rate = torchaudio.load(path)
    if sample_rate != 16000:
        audio = torchaudio.functional.resample(audio, orig_freq=sample_rate, new_freq=16000)
    input_data = processor.feature_extractor(audio[0], sampling_rate=16000, return_tensors='pt')
    for key, val in input_data.items():
        input_data[key] = val.cuda()
    # Infer
    with torch.no_grad():
        output = model(**input_data)
    logits = output[0][0].cpu().numpy()
    text = decoder.decode(
        logits, beam_width=50, token_min_logp=-10, beam_prune_logp=-10
    ).strip()
    return text


t1 = time.time()
print(f"TRANSCRIPT: {generate_transcript(args.audio_path)}")
print(f"TIME: {time.time() - t1}")