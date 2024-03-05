
from transformers import AutoProcessor, SeamlessM4TModel
processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

from huggingface_hub import notebook_login
notebook_login()

def translate(text_to_translate, src_lang, tgt_lang):
  text_inputs = processor(text = text_to_translate, src_lang=src_lang, return_tensors="pt")
  output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
  translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
  return translated_text

from google.colab import drive
drive.mount('/content/gdrive')

import chromadb
client = chromadb.PersistentClient(path="/content/gdrive/MyDrive/A04_DB")
# collection = client.get_or_create_collection(name="Stories")
collection = client.get_collection(name="Stories")
# client.list_collections()

from datasets import load_dataset

# Load model directly
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, AutoModelForTextToWaveform

# load model and processor
STTprocessor = AutoProcessor.from_pretrained("kazimAsif/whisper-STT-small-ur")
STTmodel = AutoModelForSpeechSeq2Seq.from_pretrained("kazimAsif/whisper-STT-small-ur")

TTStokenizer = AutoTokenizer.from_pretrained("kazimAsif/TTS")
TTSmodel = AutoModelForTextToWaveform.from_pretrained("kazimAsif/TTS")

# @title For Query in Audio
audio_path = "/content/Recording.m4a" # @param {type:"string"}

from pydub import AudioSegment

def convert_to_wav(audio_path):
    # Check the audio file format
    audio_format = audio_path.split('.')[-1].lower()

    if audio_format != 'wav':
        # Load the audio file
        sound = AudioSegment.from_file(audio_path, format=audio_format)

        # Change the format to WAV
        wav_path = audio_path.replace(f'.{audio_format}', '.wav')
        sound.export(wav_path, format='wav')

        return wav_path
    else:
        return audio_path

converted_audio_path = convert_to_wav(audio_path)

audio_path = converted_audio_path

import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import Audio

# Create a dummy row with a path column
dummy_row = {"audio": audio_path}
dummy_df = pd.DataFrame([dummy_row])

# Read it from pandas
dummy_dataset = Dataset.from_pandas(dummy_df)

dummy_dataset = dummy_dataset.cast_column("audio", Audio(sampling_rate=16000))

sample = dummy_dataset[0]['audio']

input_features = STTprocessor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features

# generate token ids
predicted_ids = STTmodel.generate(input_features)
# decode token ids to text
transcription = STTprocessor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
toTranslate = transcription[0]

# @title  For Query in Text
Your_Query = "برسوں بعد فرحانہ کو دیکھا تو کیا حال تھا" # @param {type:"string"}

if Your_Query =='':
  toTranslate = transcription[0]
else:
  toTranslate = Your_Query

translated_query = translate(toTranslate, 'urd', 'eng') #translate query in english
# print(translated_query)

results = collection.query(
    query_texts=[translated_query],
    n_results=3
)

# results['documents'][0]

translated_query

from getpass import getpass
import os

# REPLICATE_API_TOKEN = getpass()
os.environ["REPLICATE_API_TOKEN"] = 'r8_QKbsZIriwaXlmsmgit8sJ6fbrnAUHLR1hBdt2'

import replicate

# function to generate prompt for each document in query result of database
def process_documents(documents, translated_query):
    combined_texts = []

    for document in documents:
        pre_prompt = 'This is some text from random story. I need answer to query according to this text.' +document[0]
        prompt_input = translated_query

        iterator_output = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": f"{pre_prompt} {prompt_input} Assistant: "}
        )

        combined_text = ""
        for text in iterator_output:
            combined_text += text   # combines the output

        combined_texts.append(combined_text.strip())  # Strip to remove trailing spaces

    return combined_texts

# to generate answer for all answers

documents = results['documents'][0]
processed_texts = process_documents(documents, translated_query)

# to generate answer for only first answer

documents = results['documents'][0][:1].copy()  # getting first document in a new array
processed_texts = process_documents(documents, translated_query)

def translate_Answer_texts_to_urdu(processed_texts):
    translated_texts_urdu = []

    for text in processed_texts:
        translated_urdu = translate(text, 'eng', 'urd')
        if translated_urdu:
            translated_texts_urdu.append(translated_urdu)

    return translated_texts_urdu

translated_processed_texts_urdu = translate_Answer_texts_to_urdu(processed_texts)

# Display the translated combined texts in Urdu
for i, translated_text in enumerate(translated_processed_texts_urdu, start=1):
    print(f"Translated Answer Text {i} (Urdu): {translated_text}")

# To listen the audio of ony first(best) answer execute this
translated_processed_texts_urdu = translated_processed_texts_urdu[:1].copy()

translated_processed_texts_urdu[0]

import torch
from IPython.display import Audio

def tts_synthesis(translated_texts_urdu, TTSmodel, TTStokenizer):
    audio_list = []

    for text in translated_texts_urdu:
        inputs = TTStokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = TTSmodel(**inputs).waveform # get waveform from model

        audio_list.append(output) # add the audio in list

    return audio_list

# Use the function to perform TTS synthesis for each translated text
audio_list = tts_synthesis(translated_processed_texts_urdu, TTSmodel, TTStokenizer)

# Display the audio using IPython's Audio module
for i, audio_output in enumerate(audio_list, start=1):
    display(Audio(audio_output.numpy(), rate=TTSmodel.config.sampling_rate))
