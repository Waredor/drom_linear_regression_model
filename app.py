import requests
import streamlit as st


st.title("Расчет цены автомобиля Toyota Prius XV30 рестайлинг по городу продажи и пробегу")

with st.form("Заполнить данные об авто"):
    odo = st.number_input("Пробег автомобиля", min_value=0)
    city = st.selectbox("Укажите город продажи", ("Хабаровск", "Владивосток", "Благовещенск"))
    year = st.number_input("Год выпуска авто", min_value=2011)
    submit = st.form_submit_button("Рассчитать стоимость авто")

if submit:
    data = {
        "odo": odo,
        "khv": 1 if city == "Хабаровск" else 0,
        "vdk": 1 if city == "Владивосток" else 0,
        "blg": 1 if city == "Благовещенск" else 0,
        "year": year
    }
    response = requests.post("http://127.0.0.1:8000/predict_price", json=data)
    st.write(f'Цена авто: {response.json()["odo"]} рублей')