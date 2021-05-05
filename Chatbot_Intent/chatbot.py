import json 
import random 
import pickle 
import numpy as np 
import nltk 
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import speech_recognition as sr
import pyttsx3
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
intents = json.loads(open('intent.json').read())

model = load_model('model_chatbot.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):

    sentence_words = clean_up_sentence(sentence)

    bag = [0] * len(words)

    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1],reverse = True)
    return_list = []

    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def getResponse(ints, intents_json):

    tag = ints[0]['intent']

    list_of_intents = intents_json['intents']

    for i in list_of_intents:

        if(i['intent']== tag):

            result = random.choice(i['responses'])

            break
    return result

def talk(text):
    engine.say(text)
    engine.runAndWait()

my_voice_list = []
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice',voices[0].id)

while True:
    try: 
        with sr.Microphone() as source:

            listener.adjust_for_ambient_noise(source,duration=1)
            print('listening...')
            voice = listener.listen(source)
            command =listener.recognize_google(voice)
            command = command.lower()
            
            if command =="exit":
                ints = "GoodBye"
                response = getResponse(ints,intents)
                talk(response)
                print(f"Chatbot: {response}")
                break
                
            print(f"User:{command}")
            ints = predict_class(command)
            res = getResponse(ints,intents)
            speech = talk(res)

            print(f"Chatbot: {res}")
    except:
        pass