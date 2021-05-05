import numpy as np
class MaxlenDecide:
    def __init__(self,answer,question,max_len_question,max_len_answer):
        self.max_len_answer = max_len_answer 
        self.max_len_question = max_len_question
        self.answer = answer 
        self.question = question
    def info_about_maxlen(self):
        """
        [Learn maxlen numbers and  word count  mean by question and answer]
        """
        number_ques = []
        for i in range(len(self.question)):
            number_ques.append(len( self.question[i].split()))  
        number_ans = []
        for i in range(len(self.answer)):
            number_ans.append(len(self.answer[i].split()))  
            
        print("*"*40)
        print("Question Max Length Mean: {}".format(round(np.mean(number_ques),2)))
        print("Answer Max Length Mean: {}".format(round(np.mean(number_ans),2)))

        print("Question Max Length : {}".format(max(number_ques)))
        print("Answer Max Length : {}".format(max(number_ans)))
        print("*"*40)

    def decide_max_len(self):
        # Question and answer word count mean for each row so small 
        # That's way we don't need to take maxlen 225 or 203 we can take more small.
        # We need to do a few tries to decide on maxlen
        cnt_q = 0 
        for i in self.question:
            if len(i.split()) <= self.max_len_question:
                cnt_q += 1 
        cnt_a = 0 
        for i in self.answer:
            if len(i.split()) <= self.max_len_answer:
                cnt_a += 1 
                
        print("Question max length rate with percent: {}".format(cnt_a / len(self.question)))
        print("Answer max length rate with percent: {}".format(cnt_q / len(self.answer)))

    def trimming_with_max_len(self):
        # Now we need to trimming with max_len
        trimming_ques  = []
        trimming_ans = []
        for i in range(len(self.question)):
            if len(self.question[i].split()) < self.max_len_question and len(self.answer[i].split()) < self.max_len_answer:
                trimming_ques.append(self.question[i])
                trimming_ans.append(self.answer[i])
        print("Total Loss: {}".format((((len(self.question)-len(trimming_ques))*100) /len(self.question))))
        return trimming_ans,trimming_ques