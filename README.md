# Cantonese (zh-HK) Text Transcription using Transformers

Lab assignment 2 of ID2223 Scalable Machine Learning and Deep Learning course at KTH Royal Institute of Technology.

## Fine-Tune a pre-trained transformer model and build a serverless UI for using that model
We used [Whisper](https://openai.com/blog/whisper/) from OpenAI as the pre-trained transformer model, a Hongkong Cantonese dataset in [common_voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/sv-SE/train) was used to train a Cantonese Whisper. Training parameters are set to be the same as in [this notebook](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb) and the training was performed on Google Colab.

A basic Gradio UI on huggingface space with an inference pipeline to this model can be found [here](https://huggingface.co/spaces/tilos/id2223_lab2)

This Word Error Rate (WER) after 4000 steps of training was 61.22.

## Use data-centric approach to fine-tune the Cantonese Whisper model. 
New data sources comes from [Magic Data](https://magichub.com/datasets/guangzhou-cantonese-scripted-speech-corpus-daily-use-sentence/), which is segmented and uploaded to [huggingface dataset](https://huggingface.co/datasets/tilos/cantonese_daily) under [CC BY-NC-ND 4.0 license](https://creativecommons.org/licenses/by-nc-nd/4.0/), preprocessed by a refactorized feature pipeline [Feature Pipeline Guangzhou.py](https://github.com/Tilosmsh/IL2223_lab2/blob/main/whisper-feature-pipeline_guangzhou.py) on [Modal](https://modal.com) (Because running feature extraction with CPU on Modal is super fast and cheap ;-), barely used our $30 credit) and
stored again as huggingface dataset [Cantonese_processed_guangzhou](https://huggingface.co/datasets/tilos/cantonese_processed_guangzhou).

We encountered some problems searching for datasets for Cantonese Chinese, the second most used dialect of Chinese. [The first being Mandarin Chinese, often being referred to as, Chinese] It turned out that there is only a limited number of data sources. The writing system of Cantonese Chinese is rather fragmented, with HK & Macau using traditional characters and wording formalities and China Mainland & Malaysia using simplified characters and their own wording formalities, resulting in an inconsistency in the transcripts. In order to be consistent with the dataset used in the first task, the problem is solved by brutally forcing all characters to be preprocessed as traditional characters in the feature pipeline. Also, there is virtually no other Cantonese dataset on Hugging face, and the dataset from magic data would be the only comprehensive and formal enough Cantonese voice transcripted dataset capable of better training our model. 

Another problem we faced is data storage. We wanted to store the extracted features on Hopsworks in the beginning, yet we were unable to maintain the datatype consistency after extraction and Hopsworks do not accept inconsistent datatype as stored features. Therefore, we used huggingface dataset to store our features. 

For the second training, the parameters used can be found in [Training Pipeline Guangzhou](https://github.com/Tilosmsh/IL2223_lab2/blob/main/Training_pipeline_guangzhou.ipynb), essentially unchanged apart from a lower learning rate. The training is performed on Google Colab (We did not use Modal here because Modal GPU time is expensive :<).

This data-centric approach reduced the Word Error Rate (WER) to 60.46 using 1600 additional steps.

## Refactor the program into a feature engineering pipeline, training pipeline, and an inference program (Hugging Face Space), to improve efficiency and scalabiliy.

The refactored scripts can be found below:
- [Training Pipeline Guangzhou](https://github.com/Tilosmsh/IL2223_lab2/blob/main/Training_pipeline_guangzhou.ipynb)
- [Feature Pipeline Guangzhou](https://github.com/Tilosmsh/IL2223_lab2/blob/main/whisper-feature-pipeline_guangzhou.py)

Here is the UI, which you can generate zh-HK subtitle from audio file, video file, Youtube URL, and your microphone.
- [Whisper app on huggingface](https://huggingface.co/spaces/Chenzhou/Whisper-zh-HK)

### Authors

- [@Ziyou Li](https://www.github.com/Tilosmsh)
- [@Chenzhou Huang](https://github.com/Chenzhou98)


### Acknowledgements

 - [Github Page](https://github.com/Tilosmsh/IL2223_lab2)
 - [ID2223 @ KTH](https://id2223kth.github.io/)    
 - [Modal](https://modal.com)
 - [Hugging Face](https://huggingface.com)
 - [Google Colab](https://colab.research.google.com)
 

