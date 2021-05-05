from tensorflow.keras.layers import Input,LSTM,Embedding,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Seq2SeqModel:
    def __init__(self,embedding_dim,latent_dim,vocab_size):
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

    def encoder(self):
        enc_inputs = Input(shape= (None,))
        enc_embedding = Embedding(self.vocab_size,self.embedding_dim,mask_zero=True)(enc_inputs)
        _,state_h,state_c = LSTM(self.latent_dim,return_state=True)(enc_embedding)     
        self.enc_states = [state_h,state_c] 

        return enc_inputs
    
    def decoder(self):
        dec_inputs = Input(shape=(None,))
        self.dec_embedding = Embedding(self.vocab_size,self.embedding_dim,mask_zero=True)(dec_inputs)
        self.dec_lstm = LSTM(self.latent_dim,return_sequences=True,return_state=True)
        dec_outputs,_,_ = self.dec_lstm(self.dec_embedding,initial_state= self.enc_states)
        self.dec_dense = Dense(self.vocab_size,activation = 'softmax')
        output = self.dec_dense(dec_outputs)

        return dec_inputs,output
    
    @staticmethod
    def create_model(enc_inputs,dec_inputs,output):
        model = Model([enc_inputs,dec_inputs],output)
        print(model.summary())
        return model

    def make_inference_models(self,enc_inputs,dec_inputs):
        print(self.vocab_size)
        print(self.max_len_questions)
        dec_state_input_h = Input(shape=(self.latent_dim,))
        dec_state_input_c = Input(shape=(self.latent_dim,))
        
        dec_state_input = [dec_state_input_h,dec_state_input_c]
        dec_outputs,state_h,state_c = self.dec_lstm(self.dec_embedding,initial_state=dec_state_input)

        dec_states = [state_h,state_c]
        dec_outputs = self.dec_dense(dec_outputs)
        dec_model = Model(
            inputs=[self.dec_inputs] + dec_state_input,
            outputs=[self.dec_outputs] + dec_states)
        
        enc_model = Model(inputs=self.enc_inputs,outputs = self.enc_states)

        print("Inference Encoder Model")
        enc_model.summary()
        print()
        print("Inference Decoder Model")
        dec_model.summary()

        return enc_model,dec_model
