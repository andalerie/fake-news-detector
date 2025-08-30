import warnings
warnings.filterwarnings("ignore", message=".*longdouble.*") # вылезал ворнинг, костылем убрала

import sys, joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# подготовка NLTK
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',    quiet=True)

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


if len(sys.argv) < 2:
    print("Fail")
    sys.exit(1)

text = sys.argv[1]

# загрузка сохраненной ранее модели
model = joblib.load('models/model_fast.pkl')
vect  = joblib.load('models/vect_fast.pkl')

def predict_label(text: str):
    clean = _preprocess(text)
    vec   = _vect.transform([clean])
    label = _model.predict(vec)[0]
    prob  = _model.predict_proba(vec)[0][label]
    return label, prob

# Если запущен как скрипт — работает из командной строки
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py \"Ваш текст\"")
        sys.exit(1)
    lbl, pr = predict_label(sys.argv[1])
    print("FAKE" if lbl else "REAL", f"({pr:.2%})")