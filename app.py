import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') # stop words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# modified: menghapus not dari stop words
stop_words_modified = stopwords.words('english')
stop_words_modified.extend(['br'])
stop_words_modified.remove('not')
stop_words_modified=set(stop_words_modified)

def remove_special_chars(content):
  return re.sub(r'[^a-zA-Z0-9 ]', '', content)

# untuk stop words modified
def contraction_expansion(content):
  content = re.sub(r"won\'t", "will not", content)
  content = re.sub(r"can\'t", "can not", content)
  content = re.sub(r"shan\'t", "shall not", content)
  content = re.sub(r"n\'t", " not", content)
  return content

def remove_stop_words_modified(content):
  clean_data = []
  for i in content.split():
    word = i.strip().lower()
    if word not in stop_words_modified and word.isalpha():
      clean_data.append(word)
  return " ".join(clean_data)

def data_cleaning_stop_words_modified(content):
  content = remove_special_chars(content)
  content = contraction_expansion(content)
  content = remove_stop_words_modified(content)
  return content

tfidf = joblib.load("tfidf_final.pkl")
rf_classifier = joblib.load("rf_classifier_final.pkl")

st.title("Movie Review Sentiment Analysis Model")
st.caption("Dibuat oleh Kelompok 10 kelas 4PTI1 untuk Ujian Akhir Semester Kecerdasan Buatan")
st.markdown('''This model uses the Random Forest algorithm to classify reviews into one of two categories: positive or negative.
Try it out by inputting your own review below! :D''') 

review = st.text_input(label="", label_visibility="collapsed", placeholder="Write your review here, then press Enter!")

if review:
  clean_review = [data_cleaning_stop_words_modified(review)]
  new_tfidf = tfidf.transform(clean_review)
  prediction = rf_classifier.predict(new_tfidf)
  if prediction[0] == 0:
    sentiment = "Negative"
  else:
    sentiment = "Positive"

  st.text(f"Prediction: {sentiment}")

st.caption('''Something to note: As of now, this model only works for reviews written in English.''')