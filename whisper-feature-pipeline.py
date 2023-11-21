import os
import modal

LOCAL=False

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","datasets", "huggingface_hub", "joblib","seaborn","scikit-learn==0.24.2","dataframe-image","librosa","transformers", "torchaudio<0.12"]).apt_install(["libsndfile1"])
   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"), timeout=5000)
   def f():
       g()

def g():
    import numpy as np
    import pandas as pd
    from datasets import load_dataset, Audio
    from huggingface_hub import login, notebook_login
    from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor

    # Predefine dataset preparation function
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors='pt').input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    # Login to huggingface
    login(token="hf_*")
    notebook_login()

    # Create and load dataset (mozilla)
    cantonese  = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train+validation+test", use_auth_token=True)
    cantonese = cantonese.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

    # Load feature extractor

    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained("tilos/whisper-small-feature-extractor")
        print("Using own feature extractor for whisper small")
    except:
        feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
        print("Using openai pretrained feature extractor for whisper small")

    # Load feature tokenizer

    try:
        tokenizer = WhisperTokenizer.from_pretrained("tilos/whisper-small-feature-tokenizer", language="Chinese", task="transcribe")
        print("Using own feature tokenizer for whisper small")
    except:
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
        print("Using openai pretrained feature tokenizer for whisper small")

    # Combine to Create A WhisperProcessor

    try:
        processor = WhisperProcessor.from_pretrained("tilos/whisper-small-processor", language="Chinese", task="transcribe")
        print("Using own feature processor for whisper small")
    except:
        processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Chinese", task="transcribe")
        print("Using openai pretrained feature processor for whisper small")

    # Downsample to 16khz
    cantonese = cantonese.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare data
    cantonese = cantonese.map(prepare_dataset, num_proc=4)


    cantonese_dropped = cantonese.remove_columns(["audio", "sentence"])


    # whisper_train = fs.get_or_create_feature_group(
    #     name="whisper_feature_zh_hk",
    #     version=1,
    #     primary_key=["input_features"],
    #     online_enabled = True,
    #     description="Cantonese audio and sentences for training the whisper model"
    # )

    # cantonese_pandas = cantonese_dropped.to_pandas(batched=True, batch_size=1000) # cantonese_pandas is a generator
    # dataset_api = project.get_dataset_api()
    # for i, frame in enumerate(cantonese_pandas): # cantonese_pandas is a generator
    #     frame.to_csv('cantonese_pandas_frame_'+str(i)+'.csv')
    #     uploaded_file_path = dataset_api.upload('cantonese_pandas_frame_'+str(i)+'.csv', "Resources")
        
    # cantonese_pandas = cantonese_dropped.to_pandas(batched=True, batch_size=1000) # cantonese_pandas is a generator
    # for frame in cantonese_pandas: # cantonese_pandas is a generator
    #   whisper_train.insert(frame, write_options={"wait_for_job" : True})

    # cantonese_pandas = cantonese_dropped.to_pandas() # cantonese_pandas is a generator
    # dataset_api = project.get_dataset_api()
    # cantonese_pandas.to_csv('cantonese_pandas.csv')
    # uploaded_file_path = dataset_api.upload('cantonese_pandas.csv', "Resources")
    
    cantonese_dropped.push_to_hub("tilos/cantonese_processed")
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
