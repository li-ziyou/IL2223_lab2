import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn==0.24.2"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("ScalableML_lab1"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperTokenizer
    from transformers import WhisperProcessor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from datasets import Audio


    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login(api_key_value="CDqcnm3gyfxjyCO8.TZwOClLOwCqDp33vX0P5Q2nsvNNyEhfBMArwNoPjnb9tUSSKq6I8X35HQ5D2tlJ7")
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()
    dataset_api = project.get_dataset_api()
    downloaded_file_path = dataset_api.download(overwrite=True, path="Resources/cantonese_pandas_frame_1.csv")  #download to local

    from datasets import load_dataset
    common_voice = load_dataset("csv",data_files="D:\Github\Deep Learning\IL2223_lab2\cantonese_pandas_frame_1.csv")  #[?]file path

    cantonese_voice_train, cantonese_voice_test = common_voice, common_voice #[?] train and test
    # You can read training data, randomly split into train/test sets of features (X) and labels (y)
    #X_train, y_train, X_test, y_test = feature_view.get_train_test_split(training_dataset_version=1)

    from transformers import WhisperProcessor
    processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="zh-HK", task="transcribe")

    import torch
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    #Define a Data collator
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lengths and need different padding methods
            # first treat the audio inputs by simply returning torch tensors
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            # get the tokenized label sequences
            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad the labels to max length
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    #Elvaluation matrics
    import evaluate
    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="zh-HK", task="transcribe")
    metric = evaluate.load("wer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    #Load a pre-trained checkpoint
    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    #Define the training configuration
    from transformers import Seq2SeqTrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-zh-HK",  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )
    #Forward the training arguments to huggingface
    from transformers import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=cantonese_voice_train, #["train"],
        eval_dataset=cantonese_voice_test,   #["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(training_args.output_dir)

    #Training start
    trainer.train()
    kwargs = {
        "dataset_tags": "mozilla-foundation/common_voice_11_0",
        "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
        "dataset_args": "config: zh-HK, split: test",
        "language": "zh-HK",
        "model_name": "Whisper Small zh-HK - Ziyou Li",  # a 'pretty' name for our model
        "finetuned_from": "openai/whisper-small",
        "tasks": "automatic-speech-recognition",
        "tags": "hf-asr-leaderboard",
    }
    trainer.push_to_hub(**kwargs)



"""
    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'titanic_model' directory will be saved to the model registry. Create the dir, first.
    model_dir = "whisper_modal"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/whisper_modal.pkl")
    #fig.savefig(model_dir + "/confusion_matrix.png")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    whisper_model = mr.python.create_model(
        name="whisper_modal",
        metrics={"wer": metric['accuracy']},
        model_schema=model_schema,
        description="whisper Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    whisper_model.save(model_dir)
"""

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
