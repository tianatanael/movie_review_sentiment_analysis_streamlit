import re
from nltk.corpus import stopwords

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