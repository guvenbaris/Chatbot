from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Tokenization:
    def __init__(self,answers,questions,max_len_questions,max_len_answers):

        self.answer = answers
        self.question = questions
        self.max_len_questions = max_len_questions
        self.max_len_answers = max_len_answers
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.answer+ self.question)
        self.vocab_size =len(self.tokenizer.word_index) + 1
    def encoder_data(self):
        tokenized_questions = self.tokenizer.texts_to_sequences(self.question)
        enc_input_data = pad_sequences(tokenized_questions,
                                        maxlen=self.max_len_questions,
                                        padding='post')
        return enc_input_data
    def decoder_data(self):
        tokenized_answers = self.tokenizer.texts_to_sequences(self.answer)
        dec_input_data = pad_sequences(tokenized_answers,
                                        maxlen= self.max_len_answers,
                                        padding='post')
        for i in range(len(dec_input_data)):
            tokenized_answers[i] = tokenized_answers[i][1:]
        dec_seq = pad_sequences(tokenized_answers,
                                maxlen=self.max_len_answers,
                                padding='post')
        dec_output_data = to_categorical(dec_seq,self.vocab_size)
        return dec_input_data,dec_output_data

    def str_to_tokens(self,sentence:str):
        words = sentence.lower().split()
        tokens_list = []
        for current_word in words:
            result = self.tokenizer.word_index.get(current_word,'')
            if result != '':
                tokens_list.append(result)

        return pad_sequences([tokens_list],maxlen = self.max_len_questions,padding='post')