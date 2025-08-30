import os, pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# загрузить NLTK-ресурсы один раз
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english')) # список стоп-слов (in, to, and..) для очищения текста
lem  = WordNetLemmatizer() # для лемматизации слов: running, ran, runs => run

def text_prepare(text: str) -> str:
    text = text.lower()
    cleaned = []

    for c in text:
        if c.isalpha() or c.isspace():
            cleaned.append(c)
        else:
            cleaned.append(' ')
    text = ''.join(cleaned)


    no_stop_words = [w for w in text.split() if w not in stop_words] # удаляем стоп-слова
    return ' '.join(lem.lemmatize(w) for w in no_stop_words) # возвращаем лемматизированные слова

# загружаем csv файл с заранее объединенным датасетом
df = pd.read_csv('data/combined_news.csv').dropna(subset=['text','label'])
print(f"Загружено {len(df)} записей") # 1 флаг, по которому мы понимаем, зависла программа или работает

# предобработка данных с помощью написаной нами функции text_prepare
df['clean'] = df['text'].apply(text_prepare)
print("Предобработка данных завершена") # 2 флаг

# разделяем данные на тестовую(test) и обучающую(train) части (test_size=0.2, значит 20% уйдут в тест)
xtr, xte, ytr, yte = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF матрица (представление текста в виде векторов) и векторизация (unigrams only, min_df=10)
vect = TfidfVectorizer(max_df=0.9, min_df=10, ngram_range=(1,1))
xtr_tfidf = vect.fit_transform(xtr)
print("Векторизация завершена:", xtr_tfidf.shape) # 3 флаг

# обучение NB
model = MultinomialNB()
model.fit(xtr_tfidf, ytr)
print("Обучение завершено") # 4 флаг

# сохраняем артефакты
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model_fast.pkl')
joblib.dump(vect,  'models/vect_fast.pkl')
print("✅ Обучение закончено, модели сохранены в ./models/") # 5 флаг == КОНЕЦ
