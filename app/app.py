import streamlit as st
import pandas as pd
import pickle
from datetime import date
import numpy as np
import joblib
#CSS style------------------------
st.markdown("""
<style>
/* ----- GLOBAL ----- */
body, .stApp {
    background-color: #000 !important;
    color: #fff !important;
}

/* ----- SIDEBAR ----- */
[data-testid="stSidebar"] {
    background-color: #000 !important;
    color: #fff !important;
}

/* ----- SELECTBOX, INPUTS ----- */
div[data-baseweb="select"] > div {
    background-color: #000 !important;
    color: #fff !important;
}

div[data-baseweb="select"] svg {
    fill: #fff !important;
}

[data-baseweb="input"] {
    background-color: #000 !important;
    color: #fff !important;
    border-color: #555 !important;
}

/* ----- BUTTON ----- */
.stButton>button {
    background-color: #000 !important;
    color: #fff !important;
    border: 1px solid #fff !important;
    border-radius: 6px;
    padding: 8px 20px;
}

.stButton>button:hover {
    background-color: #111 !important;
}

/* ----- TABLES (st.dataframe, st.table) ----- */
[data-testid="stTable"] {
    background-color: #000 !important;
}

[data-testid="stTable"] table {
    background-color: #000 !important;
    color: #fff !important;
}

[data-testid="stTable"] th {
    background-color: #111 !important;
    color: #fff !important;
}

[data-testid="stTable"] td {
    background-color: #000 !important;
    color: #fff !important;
}

/* st.dataframe (interactive grid) */
.stDataFrame div[data-testid="stDataFrame"] {
    background-color: #000 !important;
}

.stDataFrame div {
    color: #fff !important;
}

.stDataFrame thead tr th {
    background-color: #111 !important;
    color: #fff !important;
}

.stDataFrame tbody tr td {
    background-color: #000 !important;
    color: #fff !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# 1. Загрузка данных для selectbox
# -------------------------------
#df = pd.read_csv('../data/processed/roud_border_tabel_1.csv')
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # путь к app/
csv_path = os.path.join(BASE_DIR, '..', 'data', 'processed', 'roud_border_tabel_1.csv')

# конвертируем путь в нормальный вид
csv_path = os.path.normpath(csv_path)

df = pd.read_csv(csv_path)

st.title("🚦 Border Traffic Prediction ")

# -------------------------------
# 2. Load trained model
# -------------------------------
@st.cache_resource
def load_model():
    # путь к текущему файлу app.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # путь к модели
    model_path = os.path.join(BASE_DIR, 'border_model.pkl')
    model_path = os.path.normpath(model_path)  # нормализуем путь
    # загружаем модель
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
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
typ_transportu = 'Razem'
#-----The average user does not need this choice.-----------------
#typ_labels = {'MRG': 'Międzynarodowy Ruch Graniczny', 'Paszportowy':'Osoby z paszportem, kontrola paszportowa',
#              'Pozasystemowa':'Osoby/przesyłki poza systemem ewidencji', 'Inny': 'Inny',
#              'Os. w INNYCH': 'Osoby w innych kategoriach/lokalizacjach', 'Razem': 'Razem'}
#typ_reverse = {v:k for k,v in typ_labels.items()}
#typ_display = st.selectbox(
#    'Select Typ',
#    ['Międzynarodowy Ruch Graniczny', 'Osoby z paszportem, kontrola paszportowa',
#     'Osoby/przesyłki poza systemem ewidencji', 'Inny', 'Osoby w innych kategoriach/lokalizacjach', 'Razem' ]
#)
#typ_transportu = typ_reverse[typ_display]
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
