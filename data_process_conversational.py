"""
Data processor for data from MagicData, and uploader to hugging face
"""
import os
import ast
import csv
from pydub import AudioSegment
from huggingface_hub import login, notebook_login
from datasets import load_dataset

path_dataset = "/Users/tilos/Downloads/scalable_ml_dl/IL2223_lab2/ASR-CCANTCSC"
path_corpus = "/Users/tilos/Downloads/scalable_ml_dl/IL2223_lab2/Guangzhou_Cantonese_Conversational_Speech_Corpus"

f_csv = open(path_dataset+'/metadata.csv', 'w')
writer = csv.DictWriter(f_csv, fieldnames = ["file_name","label"])

txt_files= os.listdir(path_corpus+"/TXT")


for file in txt_files:
    sound_file = AudioSegment.from_wav(path_corpus+"/WAV/"+file[:-4]+".wav")

    with open("/Users/tilos/Downloads/scalable_ml_dl/IL2223_lab2/Guangzhou_Cantonese_Conversational_Speech_Corpus"+"/TXT/"+file) as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            l = line.split()

            start_time, end_time= ast.literal_eval(l[0])

            start_time = int(start_time) * 1000
            end_time = int(end_time) * 1000

            label = l[-1]

            sound_slice = sound_file[start_time: end_time]

            sound_slice.export(path_dataset+'/data/'+ file[:-4] + "_" + str(i) + ".wav", format="wav")

            writer.writerow({'file_name': file[:-4] + "_" + str(i) +".wav", 'label': label})

        f.close()
    



