import warnings
warnings.filterwarnings("ignore", message=".*longdouble.*")

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# >>> Начало блока предобработки (копируется из train_fast.py) <<<

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',    quiet=True)

STOP = set(stopwords.words('english'))
LEM  = WordNetLemmatizer()
RE_CLEAN = re.compile(r'[^a-zA-Z\s]')

def fast_preprocess(text: str) -> str:
    text = text.lower()
    text = RE_CLEAN.sub(' ', text)
    tokens = [tok for tok in text.split() if tok not in STOP]
    return ' '.join(LEM.lemmatize(tok) for tok in tokens)

# >>> Конец блока предобработки <<<

# 1) Загрузка исходных данных
df = pd.read_csv('data/combined_news.csv')

# 2) Применяем предобработку ко всем текстам
df['clean'] = df['text'].apply(fast_preprocess)

# 3) Загружаем модель и векторизатор
vect  = joblib.load('models/vect_fast.pkl')
model = joblib.load('models/model_fast.pkl')

# 4) Разбиение на train/test с тем же random_state
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

# 5) Векторизация тестовых текстов
X_test_tfidf = vect.transform(X_test)

# 6) Получаем вероятности класса «Fake»
probs = model.predict_proba(X_test_tfidf)[:, 1]

# 7) Строим ROC-кривую
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print(f"ROC AUC: {roc_auc:.4f}")
