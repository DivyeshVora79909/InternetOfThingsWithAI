# bart-large-mnli
import time
start_time = time.time()
import serial 
serBT = serial.Serial('COM4', 9600)
import vosk
import pyaudio
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3

def modelAudioToText()->None:
    speech = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
    recognizer = vosk.KaldiRecognizer(speech, 16000)
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

    print("Listening ...")
    stream.start_stream()
    result = ""
    for _ in range(0, int(16000 / 8000 * 3)):     # Change the loop duration to 5 seconds
        data = stream.read(8000)
        if recognizer.AcceptWaveform(data):
            result += recognizer.Result()
    parsed_result = json.loads(result)
    recognized_text = parsed_result['text']
    print("Recognized text:", recognized_text)

    stream.stop_stream()
    stream.close()
    audio.terminate()

def modelFullBuilt0()->None:

    def audio_queuer() -> str:
        speech = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
        recognizer = vosk.KaldiRecognizer(speech, 16000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

        print("Listening ...")
        stream.start_stream()
        result = ""
        for _ in range(0, int(16000 / 8000 * 4)):     # Change the loop duration to 5 seconds
            data = stream.read(8000)
            if recognizer.AcceptWaveform(data):
                result += recognizer.Result()
        parsed_result = json.loads(result)
        recognized_text = parsed_result['text']
        print("Recognized text:", recognized_text)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        return recognized_text

    sentence = audio_queuer()
    sentences = ["lights camera and action", sentence]
    model_name_or_path = "C:/MachineLearning/HuggingFace/HuggingFaceModels/bioasq-1m-msmarco-distilbert-gpl" 
    tokenizer, modelM = AutoTokenizer.from_pretrained(model_name_or_path), AutoModel.from_pretrained(model_name_or_path)
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    with torch.no_grad():
        model_output = modelM(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_outputs = modelM(**encoded_inputs)
    embeddings = mean_pooling(model_outputs, encoded_inputs['attention_mask'])
    similarities = cosine_similarity(embeddings[1:], embeddings[0].unsqueeze(0))

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech (words per minute)
    engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
    if similarities[0][0] > 0.6:
        engine.say("Successfull.")
        print(f"Successfull, on {sentence}")
        engine.runAndWait()
    else:
        engine.say("could't recognize")
        engine.runAndWait()

def modelFullBuilt1()->None :   # sentence similarity with sentiment analyzer

    def recognize_speech(duration=4):
        speech_model = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
        recognizer = vosk.KaldiRecognizer(speech_model, 16000)
        audio = pyaudio.PyAudio()
        
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        print("Listening ...")
        stream.start_stream()
        
        result = ""
        for _ in range(int(16000 / 8000 * duration)):
            data = stream.read(8000)
            if recognizer.AcceptWaveform(data):
                result += recognizer.Result()

        stream.stop_stream()
        stream.close()
        audio.terminate()

        parsed_result = json.loads(result)
        recognized_text = parsed_result.get('text', '')
        print("Recognized text:", recognized_text)
        return recognized_text

    def calculate_similarity(sentence):
        model_name_or_path = "C:/MachineLearning/HuggingFace/HuggingFaceModels/bioasq-1m-msmarco-distilbert-gpl" 
        tokenizer, model = AutoTokenizer.from_pretrained(model_name_or_path), AutoModel.from_pretrained(model_name_or_path)
        encoded_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)    
        sentence_embedding = model_output.last_hidden_state.mean(dim=1)
        return cosine_similarity(sentence_embedding, sentence_embedding)[0][0]

    def main():
        recognized_text = recognize_speech()
        similarity_score = calculate_similarity(recognized_text)

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        if similarity_score > 0.6:
            engine.say("Successful.")
            print(f"Successful, on {recognized_text}")
        else:
            engine.say("Unsuccessful")
        engine.runAndWait()

    if __name__ == "__main__":
        main()

def modelFullBuilt2()->None:   # fast only sentence similarity

    def audio_to_text(duration=4):
        speech = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
        recognizer = vosk.KaldiRecognizer(speech, 16000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

        print("Listening ...")
        stream.start_stream()
        result = ""
        for _ in range(int(16000 / 8000 * duration)):
            data = stream.read(8000)
            if recognizer.AcceptWaveform(data):
                result += recognizer.Result()
        parsed_result = json.loads(result)
        recognized_text = parsed_result['text']
        print("Recognized text:", recognized_text)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        return recognized_text

    def calculate_similarity(sentence):
        sentences = ["light, camera and action", sentence]
        model_name_or_path = "C:/MachineLearning/HuggingFace/HuggingFaceModels/bioasq-1m-msmarco-distilbert-gpl" 
        tokenizer, model = AutoTokenizer.from_pretrained(model_name_or_path), AutoModel.from_pretrained(model_name_or_path)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = model_output.last_hidden_state[:, 0, :]
        similarities = cosine_similarity(sentence_embeddings[1:], sentence_embeddings[0].unsqueeze(0))
        return similarities[0][0], sentence

    def main():
        recognized_text = audio_to_text()
        similarity_score, sentence = calculate_similarity(recognized_text)

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        if similarity_score > 0.6:
            engine.say("Successful.")
            print(f"Successful, on {sentence}")
            engine.runAndWait()
        else:
            engine.say("Unsuccessful")
            engine.runAndWait()

    if __name__ == "__main__":
        main()

def modelFullBuilt3()->None:   # sign(ve) + 2

    def audio_to_text(duration=3):
        speech = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
        recognizer = vosk.KaldiRecognizer(speech, 16000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

        print("Listening ...")
        stream.start_stream()
        result = ""
        for _ in range(int(16000 / 8000 * duration)):
            data = stream.read(8000)
            if recognizer.AcceptWaveform(data):
                result += recognizer.Result()
        parsed_result = json.loads(result)
        recognized_text = parsed_result['text']
        print("Recognized text:", recognized_text)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        return recognized_text

    def calculate_similarity(sentence):
        sentences = ["lights camera and action", sentence]
        model_name_or_path = "C:/MachineLearning/HuggingFace/HuggingFaceModels/bioasq-1m-msmarco-distilbert-gpl" 
        tokenizer, model = AutoTokenizer.from_pretrained(model_name_or_path), AutoModel.from_pretrained(model_name_or_path)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = model_output.last_hidden_state[:, 0, :]
        similarities = cosine_similarity(sentence_embeddings[1:], sentence_embeddings[0].unsqueeze(0))
        return similarities[0][0], sentence

    def analyze_sentiment(text):
        model_name = "C:/MachineLearning/HuggingFace/HuggingFaceModels/sentiment-roberta-large-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        encoded_text = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")
        outputs = model(**encoded_text)
        probabilities = torch.softmax(outputs.logits, dim=-1).squeeze(0).tolist()
        positive_prob = probabilities[1]  # Probability of positive sentiment

        return positive_prob

    def main():
        recognized_text = audio_to_text()
        similarity_score, sentence = calculate_similarity(recognized_text)
        sentiment_score = analyze_sentiment(recognized_text)
        print("Positive probability:", sentiment_score)

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

        if similarity_score > 0.6 and sentiment_score > 0.9:# and sentiment_score > 0.7:
            engine.say("Successful. lights, camera and action")
            print(f"Successful, because you told {sentence}")
        else:
            engine.say("Unsuccessful")

        engine.runAndWait()

    if __name__ == "__main__":
        main()

def modelFullBuilt4()->None:   # bluetooth Serial + 2

    def audio_to_text(duration=4):
        speech = vosk.Model(r"C:\Users\LENOVO\Music\vosk-model-small-en-us-0.15")
        recognizer = vosk.KaldiRecognizer(speech, 16000)
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

        print("Listening ...")
        stream.start_stream()
        result = ""
        for _ in range(int(16000 / 8000 * duration)):
            data = stream.read(8000)
            if recognizer.AcceptWaveform(data):
                result += recognizer.Result()
        parsed_result = json.loads(result)
        recognized_text = parsed_result['text']
        print("Recognized text:", recognized_text)

        stream.stop_stream()
        stream.close()
        audio.terminate()
        return recognized_text

    def calculate_similarity(sentence):
        sentences = ["light, camera and action", sentence]
        model_name_or_path = "C:/MachineLearning/HuggingFace/HuggingFaceModels/bioasq-1m-msmarco-distilbert-gpl" 
        tokenizer, model = AutoTokenizer.from_pretrained(model_name_or_path), AutoModel.from_pretrained(model_name_or_path)
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        sentence_embeddings = model_output.last_hidden_state[:, 0, :]
        similarities = cosine_similarity(sentence_embeddings[1:], sentence_embeddings[0].unsqueeze(0))
        return similarities[0][0], sentence

    def main():
        recognized_text = audio_to_text()
        similarity_score, sentence = calculate_similarity(recognized_text)

        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        if similarity_score > 0.6:
            engine.say("Successful.")
            print(f"Successful, on {sentence}")
            engine.runAndWait()
        else:
            engine.say("Unsuccessful")
            print("Unsuccessful")
            engine.runAndWait()
        message = str(int(round(similarity_score)))
        serBT.write(message.encode())
        serBT.close()

    if __name__ == "__main__":
        main()

modelFullBuilt4()
print("Execution time:", time.time() - start_time, "seconds")
#input("Press Enter to exit...")
