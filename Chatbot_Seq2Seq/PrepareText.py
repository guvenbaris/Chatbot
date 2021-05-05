import re 
class PrepareText:
    """
    This class purpose prepare txt file for training.
    """
    def get_from_txt(lines,conversations): 
        """[Read txt files and split txt files by question and answers]
        """
        exchn = []
        for conver in conversations:
            exchn.append(conver.split(' +++$+++ ')[-1][1:-1].replace("'", " ").replace(",","").split())

        diag = {}
        for line in lines:
            diag[line.split(' +++$+++ ')[0]] = line.split(' +++$+++ ')[-1]

        
        questions = []
        answers =[]
        # we take just 3000 line from conversations txt. 
        for conver in exchn[:3000]:
            if len(conver) % 2==0:
                for i in range(len(conver)):
                    if i % 2 == 0:
                        questions.append(diag[conver[i]])
                    else: 
                        answers.append(diag[conver[i]])
            else:
                for i in range(len(conver)-1):
                    if i % 2 == 0 :
                        questions.append(diag[conver[i]])
                    else:
                        answers.append(diag[conver[i]])
                        

        print("*"*40)      
        print("Question Length: ",len(questions))
        print("Answer Length: ",len(answers))

        return answers,questions

    def clean_text(text):
        """[Cleaning unwanted characters and expressions]

        Args:
            text ([string]): [Text to be cleared]

        Returns:
            [type]: [Cleared text]
        """

        txt = text.lower()

        # regular n't 
        txt = re.sub("couldn't",'could not',txt)
        txt = re.sub("can't",'can not',txt)
        txt = re.sub("won't",'will not',txt)
        txt = re.sub("n't",' not',txt)

        # regular short form
        txt  = re.sub("'ll",' will',txt)
        txt = re.sub("'m",' am',txt)
        txt = re.sub("'ve",' have',txt)
        txt = re.sub("'d",' would',txt)
        txt = re.sub("'s",' is',txt)
        txt = re.sub("'re",' are',txt)

        txt = re.sub(r"n'", "ng", txt)
        txt = re.sub(r"'bout", "about", txt)
        txt = re.sub(r"'til", "until", txt)

        txt = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", txt)

        txt = txt.strip()

        return txt

    def add_token(answer):
        """
        Add tokens to start and end.
        """
        for i in range(len(answer)):
            answer[i] = '<START> ' + answer[i] + ' <END>'
        return answer
