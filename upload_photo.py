#%%
from flask import Flask
# from flask import Blueprint
from web import senti

# from title_pipline import claTextPreProc
# from title_mse import mseTextPreProc

upload = Flask(__name__)
# upload.register_blueprint(senti, url_prefix='/sentiment_ana')
# upload = Blueprint('upload', __name__)
# import tensorflow as tf
# # 從 HDF5 檔案中載入模型
# model = tf.contrib.keras.models.load_model('title_classfication.h5')
# model = tf.contrib.keras.models.load_model('title_mse.h5')

from keras.models import load_model
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
import jieba
import tensorflow as tf
from keras.preprocessing.image import load_img
import numpy as np



def TextPreProc1(title):
    seg_list=[]
    for i in range(len(title)):###where is new_title
        seg_list.append(list(jieba.cut(title[i], cut_all=False)))
    
    token = Tokenizer(num_words=3800) 
    token.fit_on_texts(seg_list)
    x_list_seq = token.texts_to_sequences(seg_list)
    #x_test_seq = token.texts_to_sequences(x_test)
    x_list = sequence.pad_sequences(x_list_seq, maxlen=30)
    word_index = token.word_index #{'天氣':1,'氣溫':2}
    return x_list

def claimgPreProc(up_img):
    upload_img = up_img.resize((32, 32))
    upload_photo = np.asarray(upload_img, dtype=np.float32)/255.0
    upload_photo=tf.convert_to_tensor(upload_photo)
    upload_photo=np.expand_dims(upload_photo, axis=0)
    return upload_photo

def mseTextPreProc(title):
    seg_list=[]
    for i in range(len(title)):###where is new_title
        seg_list.append(list(jieba.cut(title[i], cut_all=False)))
    
    token = Tokenizer(num_words=3800) 
    token.fit_on_texts(seg_list)
    x_list_seq = token.texts_to_sequences(seg_list)
    #x_test_seq = token.texts_to_sequences(x_test)
    x_list = sequence.pad_sequences(x_list_seq, maxlen=380)
    word_index = token.word_index #{'天氣':1,'氣溫':2}
    return x_list


# 載入模型

# 載入模型
title_cla_model = load_model('title_classfication.h5')
title_mse_model = load_model('title_mse.h5')
chinese_blend_model = load_model('chinese_blend.h5')
english_blend_model = load_model('english_blend.h5')
# result = title_cla_model.predict(TextPreProc(pd.Series("我是老高")))
# result2 = title_mse_model.predict(mseTextPreProc(pd.Series("我是老高")))
#print("sucess import",str(result2),"2",str(result))

#%%
# import blend_model
# user = blend_model.input("我是老高")[0]
# # user,a,b = blend_model.input("我是老高")



# # Load the Model back from file
# import pickle
# with open( "blend_model.pkl"  , 'rb') as file:  
#     blend_model = pickle.load(file)

# if user.any():
#     user = user.reshape( user.shape[0],user.shape[1]*user.shape[2])
#     proba = blend_model.predict(user)
#     print("blend success",proba)




#%%
title_cnn_model = load_model('title_cnn.h5')
result = title_cnn_model.predict(mseTextPreProc(pd.Series("我是老高")))
print("sucess cnn result",result)

#%%




import os
basedir = os.path.dirname(os.path.realpath('__file__'))
print(basedir)
import os
basedir = os.getcwd()
print(basedir)
#basedir = os.path.abspath(os.path.dirname(__file__))
 
# @app.route('/up_photo', methods=['post'])
# def up_photo():
#   img = request.files.get('txt_photo')
#   username = request.form.get("name")
#   path = basedir+"/static/photo/"
#   file_path = path+img.filename
#   img.save(file_path)
#   print('上傳頭像成功，上傳的使用者是：'+username)
#   return render_template('index.html')


# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
 
# def allowed_file(filename):
#   return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#-*-coding:utf-8-*-
import datetime
import random
class Pic_str:
  def create_uuid(self): #生成唯一的圖片的名稱字串，防止圖片顯示時的重名問題
    nowTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S"); # 生成當前時間
    randomNum = random.randint(0, 100); # 生成的隨機整數n，其中0<=n<=100
    if randomNum <= 10:
      randomNum = str(0) + str(randomNum)
    uniqueNum = str(nowTime) + str(randomNum)
    return uniqueNum

# @app.route('/download/<string:filename>', methods=['GET'])
# def download(filename):
#   if request.method == "GET":
#     if os.path.isfile(os.path.join('upload', filename)):
#       return send_from_directory('upload', filename, as_attachment=True)
#     pass

#encoding:utf-8
#!/usr/bin/env python
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort, redirect, url_for
import time
import os

#!pip install Theano
#!pip install strUtil
#from strUtil import Pic_str
import base64

UPLOAD_FOLDER = 'upload'
#upload.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
base_dir = os.path.dirname(os.path.realpath('__file__'))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
 
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
 
 
@upload.route('/upload/<string:language>', methods=['GET'])
def upload_test(language):
  return render_template('popularity_pred.html',language = language)#up

@upload.route('/popularity_pred_res')
def popularity_pred_res():
  return render_template('popularity_pred_res.html')#up

def file(filename,title):
    print(filename,title)
    #return jsonify({"title": title, "file": filename})


@upload.route("/",methods=['GET','POST'])
def main():
  #isEnglish = 'Chinese'
  if request.method == "POST":
    isEnglish = request.form.get('language')
    if isEnglish:
      print("english",isEnglish)
    else:
      isEnglish = 'Chinese'
    return redirect(url_for('function', language=isEnglish))
  return render_template("main.html")#redirect(url_for('download', filename=new_filename))

@upload.route("/function/<string:language>", methods=['GET'])
def function(language):
    print("func english",language)
    return render_template("function.html", language=language)


 
# 上傳檔案
@upload.route('/up_photo/<string:language>', methods=['POST','GET'], strict_slashes=False)
def api_upload(language):
  file_dir = os.path.join(basedir, UPLOAD_FOLDER)
  if not os.path.exists(file_dir):
    os.makedirs(file_dir)
  f = request.files['photo']
  title = request.values.get('message') 
  if f and allowed_file(f.filename):
    fname = secure_filename(f.filename)
    print(fname,"len",len(title),"title",title)
    ext = fname.rsplit('.', 1)[1]
    new_filename = Pic_str().create_uuid() + '.' + ext
    f.save(os.path.join(file_dir, new_filename))
    #redirect(url_for('download', filename=new_filename))
    #return redirect(url_for('show_photo', filename=new_filename))
    file( new_filename,title)
    if len(title)>0:
        return redirect(url_for('aimodel', filename=new_filename, usertitle = title, language = language))
       
    else:
        print(len,len(title),"title",title)
        return jsonify({"error": 1001, "msg": "no title"})
    #return redirect(url_for('aimodel',  usertitle= title))
    #return jsonify({"success": 0, "msg": title})
            
  else:
    return jsonify({"error": 1001, "msg": "fail upload"})
 
@upload.route('/download/<string:filename>', methods=['GET'])
def download(filename):
  if request.method == "GET":
    if os.path.isfile(os.path.join('upload', filename)):
      return send_from_directory('upload', filename, as_attachment=True)
    pass
# @app.route('/ai/<string:filename,string:usertitle>', methods=['GET'])
# def aimodel(usertitle,filename):
#   if request.method == "GET":
#     result = title_pipline.TextPreProc(usertitle)
#     return jsonify({ "result":result})
#     pass

# @upload.route('/ai/<string:filename>/<string:usertitle>/<string:language>', methods=['GET'])
# def aimodel(filename,usertitle,language):
#     file_dir = os.path.join(basedir, UPLOAD_FOLDER)
    
#     if request.method == "GET":
#       if language == 'Chinese':
#         if usertitle is None:
#             pass
#         else:
#             mse_result = title_mse_model.predict(mseTextPreProc(pd.Series(usertitle)))
#             cla_result = title_cla_model.predict(TextPreProc(pd.Series(usertitle)))
#             #print(cla_result, type(cla_result))
#             max_value = max(cla_result[0])
#             cla_index = cla_result[0].tolist().index(max_value)
#         if filename is None:
#             pass
#         else:
#             image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
#             response = make_response(image_data)
#             response.headers['Content-Type'] = 'image/png'
#       else:
#         if usertitle is None:
#             pass
#         else:
#             mse_result = title_mse_model.predict(mseTextPreProc(pd.Series(usertitle)))
#             cla_result = title_cla_model.predict(TextPreProc(pd.Series(usertitle)))
#             #print(cla_result, type(cla_result))
#             max_value = max(cla_result[0])
#             cla_index = cla_result[0].tolist().index(max_value)
#         if filename is None:
#             pass
#         else:
#             image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
#             response = make_response(image_data)
#             response.headers['Content-Type'] = 'image/png'
      
#             #return response

#       return render_template('popularity_pred_res.html',cla_result = str(cla_index))#jsonify({ "mse result":str(mse_result), "cla result":str(cla_result),"photo":str(filename)})
#     else:
#         pass
@upload.route('/ai/<string:filename>/<string:usertitle>/<string:language>', methods=['GET'])
def aimodel(filename,usertitle,language):
    file_dir = os.path.join(basedir,UPLOAD_FOLDER)
    upload_photo = np.zeros((1, 32, 32, 3), dtype=int)
    cla_index = 0
    usertitle =" "
    if request.method == "GET":
        if usertitle is None or filename is None:
          pass
          # if usertitle is None:
          #   usertitle =" "
          #     #pass
          # #else:
          #     #mse_result = title_mse_model.predict(mseTextPreProc(pd.Series(usertitle)))
          #     #cla_result = title_cla_model.predict(TextPreProc(pd.Series(usertitle)))
          # if filename is None:
          #   upload_photo = np.zeros((1, 32, 32, 3), dtype=int)

          #     pass
          # else:
          #     image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
          #     response = make_response(image_data)
          #     response.headers['Content-Type'] = 'image/png'
        else:
              #return response
          #image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
          #print("typenhnihbhbi:",type(up_img))
          up_img = load_img(os.path.join(file_dir, '%s' % filename))
          #print("typenhnihbhbi:",type(up_img))
          upload_photo = claimgPreProc(up_img)
          if language == 'Chinese':
            cla_result = chinese_blend_model.predict([TextPreProc1(pd.Series(usertitle)),upload_photo])
            max_value = max(cla_result[0])
            cla_index = cla_result[0].tolist().index(max_value)
          else:
            cla_result = english_blend_model.predict([TextPreProc1(pd.Series(usertitle)),upload_photo])
            if cla_result[0][0]>0.5:
              cla_index = 1
            else:
              cla_index = 0

        return render_template('popularity_pred_res.html',cla_result = cla_index)#"mse result":str(mse_result), 
    else:
        pass
  
  
# show photo
@upload.route('/show/<string:filename>', methods=['GET'])
def show_photo(filename):
  file_dir = os.path.join(basedir, UPLOAD_FOLDER)
  if request.method == 'GET':
    if filename is None:
      pass
    else:
      image_data = open(os.path.join(file_dir, '%s' % filename), "rb").read()
      response = make_response(image_data)
      response.headers['Content-Type'] = 'image/png'
      return response
  else:
    pass
 
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
@upload.route("/sentiment")
def sentiment_ana(): #用來回應網站首頁連線的函式
    return render_template("sentiment_ana.html")

@upload.route("/submit",methods=["GET", "POST"])
def submit():#從這里定義具體的函式 回傳值均為json格式
    if request.method == "POST":
        link = request.form.get("link")
        json_info=ved_info(link)
        text=json_info.get_json()
        print(text)
        pipe=[0]*len(text)
        proba=[0]*len(text)
        for i in range (len(text)):
            pipe[i] = pipeline.transform(pd.Series(text[i]))
            proba[i] = model.predict_proba(pipe[i])[0]
        proba=np.array(proba)
        pos_sum=0
        for i in range(len(text)):
            if proba[i][1]>=0.5:
                pos_sum+=1
        neg = 1-pos_sum/len(text)
        # pos = pos_sum/len(text)
        neg = int(neg*100)
        # pos = int(pos*100)
        pos=100-neg
    return render_template("sentiment_ana_res.html",neg=neg,pos=pos)

if __name__ == '__main__':
  happy = r" ([xX;:]-?[dD)]|:-?[\)]|[;:][pP]) "
  sad = r" (:'?[/|\(]) "
  model,pipeline=load_model()
  upload.run(port=5000, debug=True)
#%%