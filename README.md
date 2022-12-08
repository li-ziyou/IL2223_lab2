# Cantonese (zh-HK) Text Transcription using Transformers

Lab assignment 2 of ID2223 Scalable Machine Learning and Deep Learning course at KTH Royal Institute of Technology.

1. Based on an existing pre-trained transformer model
[Whisper](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb), a zh-HK dataset in [common_voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/sv-SE/train)
was used to train a Cantonese Whisper.

2. Use data-centric approach to fine-tune the Cantonese Whisper model. New data sources comes from [xxx](), which is being preprocessed and
stored in [Cantonese_processed_guangzhou](https://huggingface.co/datasets/tilos/cantonese_processed_guangzhou).
This data-centric approach reduce the Word Error Rate (WER) from xxx to xxx.

3. Refactor the program into a feature engineering pipeline, training pipeline, and an inference to improve efficiency and scalabiliy.

Here is the UI, which you can generate zh-HK subtitle from audio file, video file, Youtube URL, and your microphone.
- [Whisper app on huggingface](https://huggingface.co/spaces/Chenzhou/Whisper-zh-HK)

## Authors

- [@Ziyou Li](https://www.github.com/Tilosmsh)
- [@Chenzhou Huang](https://github.com/Chenzhou98)


## Acknowledgements

 - [Github Page](https://github.com/Tilosmsh/IL2223_lab1)
 - [ID2223 @ KTH](https://id2223kth.github.io/)    
 - [Hopsworks](https://www.hopsworks.ai/)
 - [Hugging Face](huggingface.co)
 

