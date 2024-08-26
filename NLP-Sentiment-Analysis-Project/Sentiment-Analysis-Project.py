from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from warnings import filterwarnings
import tkinter as tk
from tkinter import messagebox
from sklearn.linear_model import LogisticRegression
filterwarnings('ignore')

data = pd.read_csv("train.tsv",sep = "\t")

data["Sentiment"].replace(0, value = "negatif", inplace = True)
data["Sentiment"].replace(1, value = "negatif", inplace = True)
data["Sentiment"].replace(3, value = "pozitif", inplace = True)
data["Sentiment"].replace(4, value = "pozitif", inplace = True)

data = data[(data.Sentiment == "negatif") | (data.Sentiment == "pozitif")]

data.groupby("Sentiment").count()
df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]




#buyuk-kucuk donusumu
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#noktalama işaretleri
df['text'] = df['text'].str.replace('[^\w\s]','')
#sayılar
df['text'] = df['text'].str.replace('\d','')
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#seyreklerin silinmesi
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


train_x = df["text"]
train_y = df["label"]

df.iloc[0]
train_x, test_x, train_y, test_y = model_selection.train_test_split(train_x, train_y, random_state=1)

#TEST-TRAIN
# train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"],
#                                                                    df["label"],
#                                                                     random_state = 1)
train_y[0:5]
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)
train_y[0:5]
test_y[0:5]



from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)

vectorizer.get_feature_names_out()[0:5]
x_train_count.toarray()




from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
feature_names = tf_idf_word_vectorizer.get_feature_names_out()
print(feature_names[:5])




tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
tf_idf_word_vectorizer.get_feature_names_out()[0:5]
x_train_tf_idf_word.toarray()


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizer.fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)


tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_chars_vectorizer.fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)



x_train_count = vectorizer.fit_transform(train_x)
x_test_count = vectorizer.transform(test_x)

x_train_tf_idf_word = tf_idf_word_vectorizer.fit_transform(train_x)
x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)

x_train_tf_idf_chars = tf_idf_chars_vectorizer.fit_transform(train_x)
x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)








loj = linear_model.LogisticRegression()
loj_model2 = loj.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(loj_model2,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()

print("LOJ N-GRAM TF-IDF Doğruluk Oranı:", accuracy)




loj = linear_model.LogisticRegression()
loj_model3 = loj.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(loj_model3,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()

print("LOJ CHARLEVEL Doğruluk Oranı:", accuracy)


nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()

print("Naive Bayes Count Vectors Doğruluk Oranı:", accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_word,
                                           test_y,
                                           cv = 10).mean()

print("Naive Bayes Word-Level TF-IDF Doğruluk Oranı:", accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_ngram,
                                           test_y,
                                           cv = 10).mean()

print("Naive Bayes N-GRAM TF-IDF Doğruluk Oranı:", accuracy)

nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_chars,train_y)
accuracy = model_selection.cross_val_score(nb_model,
                                           x_test_tf_idf_chars,
                                           test_y,
                                           cv = 10).mean()

print("Naive Bayes CHARLEVEL Doğruluk Oranı:", accuracy)


# loj_model
yeni_yorum = pd.Series("this film is very nice and good i like it")
# v = CountVectorizer()
# v.fit(train_x)
# yeni_yorum = v.transform(yeni_yorum)
# loj_model.predict(yeni_yorum)

from sklearn.feature_extraction.text import CountVectorizer

# Fit the vectorizer on the training data
vectorizer = CountVectorizer()
vectorizer.fit(train_x)
# # Transform the training and test data using the same vectorizer
x_train_count = vectorizer.transform(train_x)
x_test_count = vectorizer.transform(test_x)
yeni_yorum = vectorizer.transform(yeni_yorum)  # Transform the new input

# # # Train the logistic regression model
# loj_model = LogisticRegression()
# loj_model.fit(x_train_count, train_y)
from sklearn.linear_model import LogisticRegression

loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, train_y)
accuracy = model_selection.cross_val_score(loj_model,
                                           x_test_count,
                                           test_y,
                                           cv = 10).mean()

print("LOJ Count Vectors Doğruluk Oranı:", accuracy)

# # Perform prediction on the new input
prediction = loj_model.predict(yeni_yorum)
print("Cümle sınıflandırması Pozitif=1    Negatif=0")
print(prediction)

# loj1 = linear_model.LogisticRegression()
# loj_model1 = loj.fit(x_train_tf_idf_word,train_y)
# accuracy = model_selection.cross_val_score(loj_model1,
#                                            x_test_tf_idf_word,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
# loj = linear_model.LogisticRegression()
# loj_model2 = loj.fit(x_train_tf_idf_ngram,train_y)
# accuracy = model_selection.cross_val_score(loj_model2,
#                                            x_test_tf_idf_ngram,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
# loj = linear_model.LogisticRegression()
# loj_model3 = loj.fit(x_train_tf_idf_chars,train_y)
# accuracy = model_selection.cross_val_score(loj_model3,
#                                            x_test_tf_idf_chars,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("CHARLEVEL Doğruluk Oranı:", accuracy)
#
#
#


# rf = ensemble.RandomForestClassifier()
# rf_model = rf.fit(x_train_count,train_y)
# accuracy = model_selection.cross_val_score(rf_model,
#                                            x_test_count,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("Count Vectors Doğruluk Oranı:", accuracy)
# rf = ensemble.RandomForestClassifier()
# rf_model = rf.fit(x_train_tf_idf_word,train_y)
# accuracy = model_selection.cross_val_score(rf_model,
#                                            x_test_tf_idf_word,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
# rf = ensemble.RandomForestClassifier()
# rf_model = loj.fit(x_train_tf_idf_ngram,train_y)
# accuracy = model_selection.cross_val_score(rf_model,
#                                            x_test_tf_idf_ngram,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
# rf = ensemble.RandomForestClassifier()
# rf_model = loj.fit(x_train_tf_idf_chars,train_y)
# accuracy = model_selection.cross_val_score(rf_model,
#                                            x_test_tf_idf_chars,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("CHARLEVEL Doğruluk Oranı:", accuracy)
#
#
#
# xgb = xgboost.XGBClassifier()
# xgb_model = xgb.fit(x_train_count,train_y)
# accuracy = model_selection.cross_val_score(xgb_model,
#                                            x_test_count,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("Count Vectors Doğruluk Oranı:", accuracy)
# xgb = xgboost.XGBClassifier()
# xgb_model = xgb.fit(x_train_tf_idf_word,train_y)
# accuracy = model_selection.cross_val_score(xgb_model,
#                                            x_test_tf_idf_word,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("Word-Level TF-IDF Doğruluk Oranı:", accuracy)
# xgb = xgboost.XGBClassifier()
# xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)
# accuracy = model_selection.cross_val_score(xgb_model,
#                                            x_test_tf_idf_ngram,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("N-GRAM TF-IDF Doğruluk Oranı:", accuracy)
# xgb = xgboost.XGBClassifier()
# xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)
# accuracy = model_selection.cross_val_score(xgb_model,
#                                            x_test_tf_idf_chars,
#                                            test_y,
#                                            cv = 10).mean()
#
# print("CHARLEVEL Doğruluk Oranı:", accuracy)
# loj_model = LogisticRegression()
# loj_model.fit(x_train_count, train_y)


# loj = linear_model.LogisticRegression()
# loj_model = loj.fit(x_train_tf_idf_word,train_y)
# accuracy = model_selection.cross_val_score(loj_model,
#                                            x_test_tf_idf_word,
#                                            test_y,
#                                            cv = 10).mean()
# print("loj Word-Level TF-IDF Doğruluk Oranı:", accuracy)
# nb = naive_bayes.MultinomialNB()
# nb_model = nb.fit(x_train_count,train_y)
# accuracy = model_selection.cross_val_score(nb_model,
#                                            x_test_count,
#                                            test_y,
#                                            cv = 10).mean()

# print("Count Vectors Doğruluk Oranı:", accuracy)



# Kullanıcının girdiği cümlenin analizini yapma fonksiyonu
def analyze_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = " ".join([Word(word).lemmatize() for word in sentence.split()])

    sentence_vectorized = vectorizer.transform([sentence])
    prediction = loj_model.predict(sentence_vectorized)
    if prediction[0] == 1:
        return 1  # Pozitif
    else:
        return 0  # Negatif

# Arayüz fonksiyonları
def analyze_button_click():
    sentence = entry.get()
    result = analyze_sentence(sentence)
    messagebox.showinfo("Sonuç", f"Analiz sonucu: {result}")

# Arayüz oluşturma
window = tk.Tk()
window.title("Cümle Analizi")
label = tk.Label(window, text="Cümleyi girin:")
label.pack()
entry = tk.Entry(window, width=120)
entry.pack()
button = tk.Button(window, text="Analiz Et", command=analyze_button_click)
button.pack()

window.mainloop()

