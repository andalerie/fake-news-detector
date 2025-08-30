# 0) Чтобы predict.py не печатал Fail и не вываливался при импорте (костыль):
import sys
sys.argv = ["app.py", "dummy"]

import streamlit as st
from langdetect import detect, LangDetectException
from predict import text_prepare

# 1) Конфигурация страницы — первая стримлит-команда
st.set_page_config(page_title="Fake News Detector", layout="centered")

# 2) Заголовок
st.title("🔍 Fake News Detector")
st.write("Введите текст новости на английском языке и нажмите **ПРЕДСКАЗАТЬ**:")

# 4) Кэшируем загрузку модели и векторизатора
@st.cache_resource
def load_artifacts():
    import joblib
    model = joblib.load('models/model_fast.pkl')
    vect  = joblib.load('models/vect_fast.pkl')
    return model, vect

model, vect = load_artifacts()

# 5) Поле ввода
user_text = st.text_area("Текст новости:", height=200)

# 6) Кнопка и предсказание
if st.button("ПРЕДСКАЗАТЬ"):
    if not user_text.strip():
        st.warning("Пожалуйста, введите текст.")
    else:
        try:
            if detect(user_text) != 'en':
                st.error("Пожалуйста, введите текст на английском языке.")
                st.stop()
        except LangDetectException:
            st.error("Язык не определён. Пожалуйста, введите текст на английском языке.")
            st.stop()

        clean = text_prepare(user_text)
        vec   = vect.transform([clean])
        label = model.predict(vec)[0]
        prob  = model.predict_proba(vec)[0][label]
        st.success(f"{'❌ FAKE' if label else '✅ REAL'}  (Вероятность: {prob:.2%})")