import streamlit as st
import pandas as pd
import pickle
from datetime import date
import numpy as np

# -------------------------------
# 1. Загрузка данных для selectbox
# -------------------------------
df = pd.read_csv('../data/processed/roud_border_tabel_1.csv')

st.title("🚦 Border Traffic Prediction – Streamlit App")

# -------------------------------
# 2. Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    with open("border_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()
st.success("Model loaded successfully!")

# -------------------------------
# 3. User inputs
# -------------------------------
selected_date = st.date_input("Pick a date", value=date.today())
day = selected_date.day
month = selected_date.month
weekday = selected_date.weekday()

# Выбор категориальных признаков
odcinek = st.selectbox("Select Odcinek", sorted(df['Odcinek'].unique()))

# --- FILTERING TRANSITIONS FOR THE SELECTED SECTION ---
filtered_przejscie = df[df['Odcinek']==odcinek]['Przejście'].unique()
przejscie = st.selectbox("Select Przejście", sorted(filtered_przejscie))
# --- AUTOMATIC DETERMINATION OF LOCATION AND BRANCH ---
row = df[df['Przejście']==przejscie].iloc[0]
placowka = row['Placówka SG']
oddzial = row['Oddział SG']
rodzaj_przejscia = row['Rodzaj przejścia']
# Mapowanie
kto_labels = {"C": "Cudzoziemiec", "RP": "Obywatel RP"}
kto_reverse = {v: k for k, v in kto_labels.items()}
# Selectbox z ładnymi nazwami
kto_display = st.selectbox(
    "Select Kto",
    ["Cudzoziemiec", "Obywatel RP"]
)
# WARTOŚĆ DO MODELU (C lub RP)
kto = kto_reverse[kto_display]
#-------------------------------------
kierunek = st.selectbox("Select Kierunek", sorted(df['Kierunek'].unique()))
#---------------------------------------
typ_labels = {'MRG': 'Międzynarodowy Ruch Graniczny', 'Paszportowy':'Osoby z paszportem, kontrola paszportowa',
              'Pozasystemowa':'Osoby/przesyłki poza systemem ewidencji', 'Inny': 'Inny',
              'Os. w INNYCH': 'Osoby w innych kategoriach/lokalizacjach', 'Razem': 'Razem'}
typ_reverse = {v:k for k,v in typ_labels.items()}
typ_display = st.selectbox(
    'Select Typ',
    ['Międzynarodowy Ruch Graniczny', 'Osoby z paszportem, kontrola paszportowa',
     'Osoby/przesyłki poza systemem ewidencji', 'Inny', 'Osoby w innych kategoriach/lokalizacjach', 'Razem' ]
)
typ_transportu = typ_reverse[typ_display]
#typ_transportu = st.selectbox("Select Typ transportu", sorted(df['Typ transportu'].unique()))

# -------------------------------
# 4. Prepare input DataFrame
# -------------------------------
cat_features = ['Placówka SG','Przejście','Rodzaj przejścia','Odcinek','Oddział SG','Kto','Kierunek','Typ transportu']
num_cols = ['day', 'month', 'weekday']
all_cols = cat_features + num_cols  # порядок как при обучении

input_df = pd.DataFrame([[
    placowka,
    przejscie,
    rodzaj_przejscia,
    odcinek,
    oddzial,
    kto,
    kierunek,
    typ_transportu,
    day,
    month,
    weekday
]], columns=all_cols)

st.subheader("🔍 Model input")
st.write(input_df)

# -------------------------------
# 5. Predict
# -------------------------------
if st.button("🔮 Oblicz prognozę"):
    log_prediction = model.predict(input_df)[0]      # модель предсказывает логарифм
    prediction = np.exp(log_prediction)              # преобразуем обратно в число людей
    st.success(f"📈 Prognozowana liczba osób: **{prediction:.0f}**")
