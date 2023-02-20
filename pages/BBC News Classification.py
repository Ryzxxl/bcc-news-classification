import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import nltk
from keras.preprocessing.text import text_to_word_sequence
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import joblib 

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


html_temp = """
<div style ="background-color:#51E1ED;padding:13px">
<h1 style ="color:black;text-align:center;"> BBC News Classification !🍥</h1>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


pipe_lr = joblib.load(open(r"pipe_SGD.pkl","rb"))

op_dict={0:"business",1:"tech",2:"politics",3:"sport",4:"entertainment",}

def preprocessDataset(train_text):
       
    #word tokenization using text-to-word-sequence
    train_text= str(train_text)
    tokenized_train_set = text_to_word_sequence(train_text,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
        
    #stop word removal
    stop_words = set(stopwords.words('english'))
    stopwordremove = [i for i in tokenized_train_set if not i in stop_words]
        
     
    #join words into sentence
    stopwordremove_text = ' '.join(stopwordremove)
        
        
    #remove numbers
    numberremove_text = ''.join(c for c in stopwordremove_text if not c.isdigit())
       
        
    #--Stemming--
    stemmer= PorterStemmer()

    stem_input=nltk.word_tokenize(numberremove_text)
    stem_text=' '.join([stemmer.stem(word) for word in stem_input])
        
        
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)

    lem_input = nltk.word_tokenize(stem_text)
    lem_text= ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in lem_input])
        
    return lem_text

def model_Predict(text):

    results = pipe_lr.predict([text])
    return (results[0])
    



def main():
     try:
            raw_text = st.text_area("Type your Text here .....",key="text")
            col1, col2, col3 = st.columns(3)

            def clear_text():
                st.session_state["text"] = ""
                
                
            if st.button("FIND"):
                clean_text =preprocessDataset(raw_text)

                output=model_Predict(clean_text)

                final_output=op_dict[(output)]


                col2.write(final_output)
            
                st.button("REST", on_click=clear_text)


     except Exception as e:
                st.text(e)


st.sidebar.subheader("About App")
st.sidebar.text("BBC News Classification App with Streamlit")


st.sidebar.subheader("By")
st.sidebar.text("Rakshit Khajuria - 19bec109")
st.sidebar.text("Prikshit Sharma - 19bec062")

             
            
 



if __name__ == '__main__':
	main()