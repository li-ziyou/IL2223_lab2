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

    trainset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="train+validation", use_auth_token=True).to_pandas()
    testset = load_dataset("mozilla-foundation/common_voice_11_0", "zh-HK", split="test", use_auth_token=True).to_pandas()

    # Remove additional metadata information
    trainset = trainset.drop(['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'], axis=1)
    testset = testset.drop(['client_id', 'path', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'], axis=1)

    whisper_train = fs.get_or_create_feature_group(
        name="whisper_feature_zh_hk_train",
        version=1,
        primary_key=["audio"],
        online_enabled = True,
        description="Cantonese audio and sentences for training the whisper model"
    )

    whisper_test = fs.get_or_create_feature_group(
        name="whisper_feature_zh_hk_test",
        version=1,
        primary_key=["audio"],
        online_enabled = True,
        description="Cantonese audio and sentences for testing the whisper model"
    )

    whisper_train.insert(trainset, write_options={"wait_for_job" : True})
    whisper_test.insert(testset, write_options={"wait_for_job" : True})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
