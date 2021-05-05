# ChatBot

In this project, have two different chatbot. First one is created from intent file. Second one is created from movie dialog corpus file. 

# 1. Intent Chatbot Explanation

In this part our purpose is to guess which part of the question asked belongs and to be able to answer according to that part. Thus we can give closed answer to question. This is our purpose. In this project used speach recognation and text to speech conversaion. 

## 1.1. Data Preprocessing And Model Creating

First, we separated words and intents corresponding to words from the data set. We used WordNetLemmatizer for find each word stem. So we reduced our data to training, then we created a bag of words thus our training dataset is ready. We need to just split it by x_train,y_train. Now we are ready for training. In this project we used neaural network. Our dataset not very big so we used basic neural network. 
```
model = Sequential()
model.add(Dense(256, input_shape = (len(x_train[0]), ), activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation='softmax')) 
``` 
## 1.2 Chatbot Creating 
In here we used five function, first function for text cleaning, second for find bag of words counterpart, third for predict to which class have question, fourth for take the answers than predicted(third part), last function will convert text to sound.

# 2. Chatbot Seq2Seq 

In this project used sequence to sequence model [seq2seq](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html). We used [Movie Dialogue Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset but we just used part of the data set. We split four part our project. We defined each one separately class. As seen below. In addition to these, we have defined main.py so that we can use them.

* PrepareData  : Read txt files and split txt files by question and answers 
* MaxlenDecide : Shrinking the data set according to the max_len we have determined
* Tokenization : To enable us to express words with vectors and return encoder,decoder input and decoder output data
* Seq2SeqModel : Sequence to Sequence model and make inference model (for the prediction)
