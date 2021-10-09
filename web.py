from flask import Flask, render_template #載入Flask
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import joblib
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
import numpy as np
from flask import request
from crawler import ved_info
import json
# from upload_photo import upload
# senti=Flask(__name__) #建立application物件

from flask import Blueprint
senti = Blueprint('senti', __name__)
# senti.register_blueprint(upload, url_prefix='/sentiment_ana')

class TextPreProc(BaseEstimator,TransformerMixin):
    def __init__(self, use_mention=False):
        self.use_mention = use_mention
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        # We can choose between keeping the mentions
        # or deleting them
        if self.use_mention:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", " @tags ")
        else:
            X = X.str.replace(r"@[a-zA-Z0-9_]* ", "")
            
        # Keeping only the word after the #
        X = X.str.replace("#", "")
        X = X.str.replace(r"[-\.\n]", "")
        # Removing HTML garbage
        X = X.str.replace(r"&\w+;", "")
        # Removing links
        X = X.str.replace(r"https?://\S*", "")
        # replace repeated letters with only two occurences
        # heeeelllloooo => heelloo
        X = X.str.replace(r"(.)\1+", r"\1\1")
        # mark emoticons as happy or sad
        X = X.str.replace(happy, " happyemoticons ")
        X = X.str.replace(sad, " sademoticons ")
        X = X.str.lower()
        return X
        
def stem_tokenize(text):
    stemmer = SnowballStemmer("english")
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(token) for token in word_tokenize(text)]

def lemmatize_tokenize(text):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in word_tokenize(text)]

def load_model():
    model = joblib.load('sentiment_analysis_sklearn_model')
    pipeline = joblib.load('pipeline')
    return model, pipeline

#建立網站首頁的回應方式
@senti.route("/")#("/<string:language>", methods=['GET'])
def sentiment_ana(): #用來回應網站首頁連線的函式
    return render_template("sentiment_ana.html", language = 'English')

# @senti.route("/submit",methods=["GET", "POST"])
# def submit():#從這里定義具體的函式 回傳值均為json格式
#     if request.method == "POST":
#         link = request.form.get("link")
#         json_info=ved_info(link)
#         text=json_info.get_json()
#         print(text)
#         pipe=[0]*len(text)
#         proba=[0]*len(text)
#         for i in range (len(text)):
#             pipe[i] = pipeline.transform(pd.Series(text[i]))
#             proba[i] = model.predict_proba(pipe[i])[0]
#         proba=np.array(proba)
#         pos_sum=0
#         for i in range(len(text)):
#             if proba[i][1]>=0.5:
#                 pos_sum+=1
#         neg = 1-pos_sum/len(text)
#         # pos = pos_sum/len(text)
#         neg = int(neg*100)
#         # pos = int(pos*100)
#         pos=100-neg
#     return render_template("sentiment_ana_res.html",neg=neg,pos=pos)

# @senti.route("/main")
# def main():
#     return render_template("main.html")

#啟動網站伺服器
if __name__ == '__main__':  
    happy = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
    sad = r" (:'?[/|\(]) "
    model,pipeline=load_model()
    # senti.run() 
    # X = pd.read_csv("comment.csv",encoding="utf-8")

    # n=len(X["comment"])
    # id_num=X["id"][len(X["id"])-1]
    # #for j in range(id_num+1):
    # text=[]
    # for i in range(len(X["comment"])):
    #     if(X["id"][i]==0):
    #         text.append(X["comment"][i])
    
    

    
        # print("vedio #",j,":")
        # print("The number of comments is",len(text))
        # print("The percentage that this vedio's comments are sad is:", 1-pos_sum/len(text))
        # print("The percentage that this vedio's comments are happy is:", pos_sum/len(text))