import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Flop Detector", layout="wide")

#Determina la stagione di uscita
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

#Caricamento asset e dati
@st.cache_resource
def load_assets():
    model = joblib.load('../notebooks/modello_film.pkl')
    scaler = joblib.load('../notebooks/scaler.pkl')
    feature_names = model.get_booster().feature_names
    #Dataset pulito
    df_sample = pd.read_csv('../data/processed/dataset_pulito.csv')
    return model, scaler, feature_names, df_sample

model, scaler, feature_names, df_pulito = load_assets()

@st.cache_data
def get_mappings(df):
    #Con target encoding calcola nome e media successo
    dir_map = df.groupby('director')['success'].mean().to_dict()
    act_map = df.groupby('lead_actor')['success'].mean().to_dict()
    comp_map = df.groupby('main_company')['success'].mean().to_dict()
    global_mean = df['success'].mean() #Valore di default se il nome non esiste
    return dir_map, act_map, comp_map, global_mean

dir_map, act_map, comp_map, global_mean = get_mappings(df_pulito)

st.title("Flop Detector")
st.markdown("Previsione del successo di un film con dati disponibili prima della sua uscita.")

with st.sidebar:
    st.header("Parametri Tecnici")
    budget = st.number_input("Budget ($)", min_value=0, value=50000000, step=1000000)
    popularity = st.slider("Popularity (Hype)", 0.0, 500.0, 200.0) #Range coerente con TMDB
    runtime = st.number_input("Durata (min)", min_value=1, value=110)
    release_year = st.number_input("Anno di uscita", 1990, 2025, 2025)

st.subheader("Team e Strategia")
col1, col2, col3 = st.columns(3)

with col1:
    director = st.selectbox("Regista", sorted(df_pulito['director'].unique()), index = 213)
    genre = st.selectbox("Genere", sorted(df_pulito['main_genre'].unique()))

with col2:
    actor = st.selectbox("Attore Protagonista", sorted(df_pulito['lead_actor'].unique()), index=50)
    #Calcola stagione
    month = st.slider("Mese di uscita", 1, 12, 6)
    season = get_season(month)

with col3:
    company = st.selectbox("Casa di Produzione", sorted(df_pulito['main_company'].unique()), index=5)

#Logica di predizione
if st.button("Calcola probabilità di successo", use_container_width=True):

    #Inizializzazione
    input_dict = {name: 0.0 for name in feature_names}
    
    #Inserimento valori numerici
    input_dict['budget'] = budget
    input_dict['popularity'] = popularity
    input_dict['runtime'] = runtime
    input_dict['release_year'] = release_year
    input_dict['release_month'] = month
    
    #Mediane
    input_dict['cast_size'] = 15
    input_dict['n_genres'] = 2
    input_dict['n_keywords'] = 10
    input_dict['is_english'] = 1
    
    #Target Encoding
    input_dict['director_te'] = dir_map.get(director, global_mean)
    input_dict['lead_actor_te'] = act_map.get(actor, global_mean)
    input_dict['main_company_te'] = comp_map.get(company, global_mean)
    
    #One-Hot Encoding per geneere e stagione
    if f"main_genre_{genre}" in input_dict:
        input_dict[f"main_genre_{genre}"] = 1
    if f"release_season_{season}" in input_dict:
        input_dict[f"release_season_{season}"] = 1

    #Creazione DataFrame e Scaling
    input_df = pd.DataFrame([input_dict])[feature_names]
    cols_to_scale = ['budget', 'popularity', 'runtime', 'release_year', 'release_month', 'n_genres', 'n_keywords', 'cast_size']
    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])

    prob = model.predict_proba(input_df)[0][1]
    
    #Output
    st.divider()
    
    #Tachimetro con plotly
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilità di Successo %", 'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 45], 'color': "#ff4b4b"},
                {'range': [45, 65], 'color': "#ffa500"},
                {'range': [65, 100], 'color': "#00cc96"}]
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    #Messaggi finali
    if prob > 0.65:
        st.balloons()
        st.success(f"Il film ha ottime probabilità di essere un successo!.")
    elif prob > 0.45:
        st.warning("Il successo di questo film è incerto e dipenderà dal marketing e dalla concorrenza.")
    else:
        st.error("Questo film ha scarse probabilità di successo.")

#Info sul modello
with st.expander("ℹInformazioni sul Modello"):
    st.write("""
    Il modello definisce **Successo** un film che soddisfa contemporaneamente:
    - **ROI ≥ 2.0** (Il film ha incassato almeno il doppio del budget)
    - **Vote Average ≥ 6.0** (Il film è stato gradito dal pubblico)
             
    NB: Per semplicità di utilizzo, per alcune colonne (cast_size, n_genres, n_keywords, is_english) è stata utilizzata la mediana. L'obiettivo di questa demo è fornire un'idea generale del funzionamento del modello, per cui la facilità di utilizzo è stata preferita all'accuratezza.
    """)