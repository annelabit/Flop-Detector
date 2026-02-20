# 🎬 Movie Flop Detector: Previsione del Successo Cinematografico

Questo progetto di Machine Learning ha l'obiettivo di prevedere se un film sarà un successo commerciale o un flop, utilizzando esclusivamente informazioni disponibili prima dell'uscita nelle sale (budget, generi, cast, regista, stagione di rilascio, ecc.).

Il dataset utilizzato è basato sul **TMDB 5000 Movie Dataset**, integrato e ripulito per estrarre metriche strutturali e finanziarie affidabili.

## 📂 Struttura del Progetto

Il progetto è suddiviso in tre fasi logiche, ognuna documentata in un Jupyter Notebook dedicato:

1. **`EDA_Cleaning.ipynb` (Esplorazione e Pulizia)**
   - Unione dei dataset `movies` e `credits`.
   - Parsing dei dati complessi in formato JSON (estrazione di generi, attori e registi).
   - Pulizia dei valori nulli e rimozione delle anomalie (es. film con budget/incassi a zero).
   - Analisi Esplorativa (EDA) tramite visualizzazioni grafiche.

2. **`FeatureEngineering.ipynb` (Ingegnerizzazione dei Dati)**
   - Creazione della variabile target binaria (`success` vs `flop`) basata sul ROI (Return on Investment) e sui voti della critica.
   - Prevenzione del Data Leakage rimuovendo le variabili post-uscita.
   - Trasformazione delle variabili categoriche tramite *One-Hot Encoding*.
   - Suddivisione in Train e Test set (stratificata) e standardizzazione delle feature numeriche con `StandardScaler`.

3. **`Model_Training.ipynb` (Addestramento e Valutazione)**
   - Addestramento di diversi modelli di classificazione: Logistic Regression, Decision Tree e Random Forest.
   - Ottimizzazione degli iperparametri tramite `GridSearchCV`.
   - Valutazione comparativa utilizzando metriche standard (Accuracy, Precision, Recall, F1-Score) e Matrici di Confusione.

## 🚀 Come eseguire il progetto

1. Clona questo repository e spostati nella cartella principale.
2. Assicurati di avere un ambiente Python configurato.
3. Installa le librerie necessarie eseguendo da terminale:
   ```bash
   pip install -r requirements.txt
