import os
import modal


LOCAL=True

if LOCAL == False:

   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4","datasets", "huggingface_hub", "joblib","seaborn","scikit-learn==0.24.2","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"))
   def f():
       g()

def g():
    import hopsworks
    import numpy as np
    import pandas as pd
    import librosa
    from datasets import load_dataset, DatasetDict
    from huggingface_hub import login, notebook_login

    project = hopsworks.login()
    fs = project.get_feature_store()

    # Login to huggingface
    login(token="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")
    notebook_login()

    # Create and load dataset
    common_voice = DatasetDict()

    common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train+validation", use_auth_token=True)
    common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="test", use_auth_token=True)
    
    # Remove additional metadata information
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    print(common_voice)

    whisper_fg = fs.get_or_create_feature_group(
        name="whisper_feature_zh_hk",
        version=1,
        primary_key=["audio"], 
        description="Cantonese audio and sentences for training whisper model"
    )

    whisper_fg.insert(common_voice["train"].to_pandas(), write_options={"wait_for_job" : False})
    whisper_fg.insert(common_voice["test"] .to_pandas(), write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
