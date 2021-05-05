import numpy as np
from PrepareText import PrepareText
from MaxlenDecide import MaxlenDecide
from Tokenization import Tokenization
from Seq2SeqModel import  Seq2SeqModel
from tensorflow.keras.optimizers import Adam

lines = open("dataset/movie_lines.txt",encoding='utf-8',
        mode ='r',errors='ignore').read().split('\n')

conversations = open("dataset/movie_conversations.txt",encoding='utf-8',
                mode = 'r',errors='ignore').read().split('\n')

# Prepare
prepare = PrepareText
answers, questions = prepare.get_from_txt(lines,conversations)
answer_clean = [prepare.clean_text(word) for word in answers]
question_clean = [prepare.clean_text(word) for word in questions]
answer_clean = prepare.add_token(answer_clean)

# Maxlen
max_len_question = 30
max_len_answer = 35
maxlen_decide = MaxlenDecide(answer_clean,question_clean,max_len_question,max_len_answer)
maxlen_decide.info_about_maxlen(),maxlen_decide.decide_max_len() # for information how we decided maxlen
short_answer,short_question = maxlen_decide.trimming_with_max_len()

# Tokenization
token = Tokenization(answers=short_answer,
                    questions=short_question,
                    max_len_questions=max_len_question,
                    max_len_answers=max_len_answer)

encoder_input_data = token.encoder_data()
decoder_input_data,decoder_output_data = token.decoder_data()

# Model 
latent_dim = 200
epoch = 100
batch_size = 50
embedding_dim = 200

sequence = Seq2SeqModel(latent_dim,embedding_dim,token.vocab_size)
enc_inputs,enc_outputs =  sequence.encoder()
dec_inputs,dec_outputs = sequence.decoder(enc_outputs)

model = Seq2SeqModel.create_model(enc_inputs,dec_inputs,dec_outputs)

model.compile(optimizer = Adam(lr=0.002) ,loss = 'categorical_crossentropy')

model.fit([encoder_input_data, decoder_input_data],
          decoder_output_data,
          batch_size= batch_size,
          epochs=epoch,verbose=1)

model.save("new_model_2.h5")

print("Finished...")
# Infrerence Models
model.load("new_model_2.h5")
enc_model, dec_model = sequence.make_inference_models(enc_inputs,dec_inputs)

while True: 
    text = input("Enter Question: ")
    if text == 'q':
        break
    else:
        text = PrepareText.clean_text(text)
        states_values = enc_model.predict(token.str_to_tokens(text))
        empty_target_seq = np.zeros((1,1))
        empty_target_seq[0,0] = token.tokenizer.word_index['start']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs,h,c = dec_model.predict([empty_target_seq] + states_values)
            sampled_word_index = np.argmax(dec_outputs[0,-1,:])
            sampled_word = None

            for word,index in token.tokenizer.word_index.items():
                if sampled_word_index == index:
                    if word != 'end':
                        decoded_translation +=' {}'.format(word)
                    sampled_word = word
            if sampled_word == 'end' or len(decoded_translation.split()) > token.max_len_answers:
                stop_condition = True
            empty_target_seq = np.zeros((1,1,))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]
    print(decoded_translation)