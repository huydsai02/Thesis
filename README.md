
### Dependencies
To install required library
```
pip install -r requirements.txt
```

# Trainer
To train non-streaming model
```
python ./train.py --train_path "path to train dataset"
```

To train streaming model
```
python ./train.py --train_path "path to train dataset" --initial_model "path to non-streaming model" --streaming
```

# Inference
To infer non streaming model
```
python ./nonstreaming_infer.py --audio_path "path to audio" --checkpoint "path to checkpoint" --language_model_path "path to language model" --syllables_path "path to syllables"
```

To infer streaming model
```
python ./streaming_infer.py --audio_path "path to audio" --checkpoint "path to checkpoint" --language_model_path "path to language model" --syllables_path "path to syllables"
python ./streaming_infer_overlap.py --audio_path "path to audio" --checkpoint "path to checkpoint" --language_model_path "path to language model" --syllables_path "path to syllables"
```

