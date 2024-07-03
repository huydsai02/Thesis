# import pyaudio  
import wave  
import numpy as np
import sys
import time
import torch, json, os
# from new_wav2vec_joint import Wav2Vec2JointClassification
from wav2vec2_model import Wav2Vec2SmallNonStreamingForCTC
# from w2v_customized_nvlb import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import kenlm
import librosa
import pandas as pd
from jiwer import wer
from sklearn.metrics import classification_report
from tqdm import tqdm
import torchaudio
from alphabet import Alphabet
from decoder import BeamSearchDecoderCTC
from language_model import LanguageModel
from scipy.io import wavfile
import librosa, time, glob, argparse
tqdm.pandas()

parser = argparse.ArgumentParser(description = "Streaming_inference")
parser.add_argument('--checkpoint', type=str, default="", help='Path of the checkpoint')
parser.add_argument('--audio_path', type=str, default="", help='Path of the audio')
parser.add_argument('--language_model_path', type=str, default="", help='Path of the language_model')
parser.add_argument('--syllables_path', type=str, default="", help='Path of the syllables')
args = parser.parse_args()

CHECKPOINT_FOLDER = args.checkpoint
print("Checkpoint:", CHECKPOINT_FOLDER)
NUM_MEM_LENGTH = 64
NUM_FRAME   = 48
NUM_FUTURE_FRAME = 16


model = Wav2Vec2SmallNonStreamingForCTC.from_pretrained(
    CHECKPOINT_FOLDER,
    need_feature_transform=True,
    mse_hidden_states_ratio=0, #Remove Reshape layer
    use_streaming_attention_mask=False,
    mem_length=NUM_MEM_LENGTH+NUM_FUTURE_FRAME, num_frames=NUM_FRAME,
    type_positional_conv_embedding="causal"
).eval().cuda()


teacher_processor = Wav2Vec2Processor.from_pretrained("nguyenvulebinh/wav2vec2-large-vi-vlsp2020")
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side="right",
    do_normalize=False, return_attention_mask=False
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=teacher_processor.tokenizer
)

assert model.wav2vec2.training == False, "Training mode is still on"

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

FIRST_CHUNK = (NUM_FRAME - 1) * 320 + 400
OTHER_CHUNK = (NUM_FRAME - NUM_FUTURE_FRAME) * 320

def streaming_asr(path):

    _, sr = torchaudio.load(path)
    if sr != 16000:
        data, _ = librosa.load(path, sr=16000)
        path = os.path.join(os.path.dirname(__file__), "streaming_overlap_temp.wav")
        scaled = np.int16(data / np.max(np.abs(data)) * 32767)
        wavfile.write(path, 16000, scaled)
       
    f = wave.open(path,"rb")  
    
    # beams, cached_lm_scores, cached_p_lm_scores = decoder.get_starting_state()
    beams = cached_lm_scores = cached_p_lm_scores = None
    total_frames = f.getnframes()
    current_frame = 0
    mems = None
    is_end = False
    processed_frames = 0

    while not is_end:
        if current_frame == 0:
            data = f.readframes(FIRST_CHUNK)
            current_frame += FIRST_CHUNK
            signal = np.frombuffer(data, dtype='int16') * 0.5**15
            audio_input = signal
        else:
            data = f.readframes(OTHER_CHUNK)
            current_frame += OTHER_CHUNK
            signal = np.frombuffer(data, dtype='int16') * 0.5**15
            # print(audio_input.shape)
            start_at = audio_input.shape[0] - NUM_FUTURE_FRAME * 320 - 80
            audio_input = np.concatenate([audio_input, signal])[start_at:]

        if current_frame > total_frames - 400:
            is_end = True

        inp = processor.feature_extractor([audio_input], return_tensors="pt", sampling_rate=16000)
        for k in inp:
            inp[k] = inp[k].cuda()

        with torch.no_grad():
            model_output = model(
                **inp, return_dict=True,
                mems=mems
            )
        mems = model_output.mems

        for key in mems:
            mems[key] = mems[key][:, :-NUM_FUTURE_FRAME, :]
            # print("MEM:", mems[key].shape)
        
        logits = model_output.logits.cpu().detach().numpy()[0]
        if not is_end:
            logits = logits[: (NUM_FRAME - NUM_FUTURE_FRAME), :]
        # print(logits.shape)
        temp_out = decoder.partial_decode_logits(
            logits=logits,
            cached_lm_scores=cached_lm_scores,
            cached_p_lm_scores=cached_p_lm_scores,
            beams=beams,
            processed_frames=processed_frames,
            is_end=is_end,
            beam_width=50, 
            token_min_logp=-10, 
            beam_prune_logp=-10,
        )

        if not is_end:
            beams, cached_lm_scores, cached_p_lm_scores = temp_out
            processed_frames += logits.shape[0]

    f.close()
    text = temp_out[0][0]
    # print(text)
    # assert False
    return text

t1 = time.time()
print(f"TRANSCRIPT: {streaming_asr(args.audio_path)}")
print(f"TIME: {time.time() - t1}")