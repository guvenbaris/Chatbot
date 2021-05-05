# Imports 
import nltk
import pickle 
import json 
import random
import numpy as np 
from nltk.stem.wordnet import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD
nltk.download('punkt')

# Read json file
with open("intent.json") as f:
    intents = json.load(f)

# Take words,classes and document(sentence+classes)
words =[]
classes = []
documents = []
ignore_list = ['!','?',',','.']

for intent in intents["intents"]:
    for text in intent["text"]:
        words_list = nltk.word_tokenize(text)
        words.extend(words_list)
        documents.append(words_list,intent["intent"])
        if intent["intent"] not in classes:
            classes.append(intent["intent"])

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_list]

words = sorted(set(words.lower()))
classes = sorted(set(classes))

pickle.dump(words,open("words.pkl",'wb'))
pickle.dump(classes,open("classes.pkl",'wb'))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words: 
        bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag,output_row])

random.shuffle(training)
training = np.array(training,dtype='object')

x_train = list(training[:,0])
y_train = list(training[:,1])

model = Sequential()
model.add(Dense(256, input_shape = (len(x_train[0]), ), activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation='softmax'))

sgd = SGD(lr =0.01,decay = 1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss = 'categorical_crossentropy',  metrics= ['accuracy'])

model.fit(np.array(x_train),np.array(y_train), epochs= 50, batch_size=5,verbose=1)
model.save('model_chatbot.h5')
print("Done")

