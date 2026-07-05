# 🧠 Segmentazione dei tumori cerebrali

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-Medical%20AI-61b440.svg)](https://monai.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-ff4b4b.svg)](https://streamlit.io/)

---

## Overview & Problem Statement
Questo progetto consiste nello sviluppo di un'applicazione per la **segmentazione automatizzata dei gliomi e glioblastomi cerebrali** a partire da esami di risonanza magnetica (MRI). L'obiettivo principale è fornire un sistema di supporto alla decisione medica per ridurre il carico cognitivo dei radiologi durante le fasi di ispezione degli esami (con modalità come la *Cine Mode*) e minimizzare il rischio di falsi negativi dovuti a fatica visiva.

L'obietivo del progetto è quello di creare un sistema che si adatti ai vincoli di budget delle strutture sanitarie pubbliche, permettendo anche ai poliambulatori periferici di utilizzare l'applicazione senza richiedere costosi aggiornamenti di server o GPU.

Sfruttando un approccio volumetrico a singolo canale (incentrato sul canale FLAIR) è possibile ridurre la quantità di dati e la memoria VRAM richiesta per le analisi, infatti in questo modo l'analisi può essere eseguita sulla CPU di normali PC commerciali presenti di solito nei reparti.


---

## Pipeline & Architettura del Sistema
L'applicazione implementa un lavoro suddiviso in quattro stadi:
1. **Acquisizione dell'immagine e Preelaborazione con MONAI:** Prende la risonanza magnetica grezza del cervello e la "pulisce" lasciando solo la sequenza visiva in cui il tumore si vede meglio (FLAIR), taglia via tutto lo sfondo nero che non serve a nulla (Crop) e uniforma la dimensione dell'immagine. Infine, applica una normalizzazione (Z-Score) per fare in modo che le immagini abbiano la stessa luminosità e contrasto, a prescindere dall'ospedale o dal tipo di macchinario che ha fatto la risonanza.
2. **Osservare i dettagli:** Il progetto si affida al paradigma delle Learned Features. Infatti Il modello non ha ricevuto alcuna istruzione sulla forma, sulla dimensione o sulla geometria specifica del tumore ed è stata addestrata fornendole in input centinaia di risonanze magnetiche.
3. **Logica Fondamentale:** Trova la posizione esatta della massa tumorale. Utilizza la rete neurale U-Net.
Essa è composta dall'**Encoder** che rimpicciolisce l'immagine per comprendere il contesto generale, poi il **Decoder** che ricostruisce l'immagine a grandezza naturale e infine ci sono le **Skip Connections** che fungono da ''ponti'' che collegano la sinistra e la destra e aggiunge i dettagli andati persi durante le operazioni precedenti
4. **Post-Elaborazione:** Dopo che la rete neurale finisce di elaborare la risonanza, esprime un valore numerico grezzo (log-odds). Questo punteggio viene passato dentro la funzione matematica **Sigmoide**, che "schiaccia" i valori all'interno di una scala fissa che va da 0 a 1 (da 0% a 100%). Successivamente, queste probabilità vengono filtrate tramite una **soglia rigida di binarizzazione (Thresholding)**. Alla fine, viene generata la maschera booleana definitiva, dove ogni pixel riceve un verdetto netto: Tumore (valore 1) oppure Tessuto Sano (valore 0).

---

## Sintesi dei Risultati
Il modello è stato ottimizzato utilizzando la funzione **Dice Loss** accoppiata all'ottimizzatore **Adam** e l'addestramento è stato condotto per un blocco complessivo di **45 epoche**.

### Metriche Finali di Validazione 
* **Train Loss:** `0.1035`
* **Validation Loss:** `0.1420`
* **Dice Score:** **`85.80%`** (Precisione del modello)
* **Mean Intersection over Union (mIoU):** **`75.13%`**

---

## Istruzioni di Installazione e Funzionamento

### Requisiti Prerequisiti
Assicurati di avere installato sul tuo sistema **Python 3.10** (o versioni superiori) e il gestore di pacchetti `pip`.

### 1. Clonare la Repository
```bash
git clone (https://github.com/DevMarzia/RilevaTumore.git)
cd RilevaTumore
```
### 2. Installazione delle Dipendenze 
Installa tutti i moduli software necessari tramite il file dei requisiti:
```bash 
pip install -r requirements.txt
```
### 3. Esecuzione dell'Applicazione Web (Streamlit)
Per lanciare l'interfaccia utente in locale, esegui il seguente comando nel terminale:
```bash
streamlit run app.py
```
Una volta aperta la pagina web, nella sidebar potrai scegliere quale risonanza visionare tra quelle usate per l'addestramento e quelle ignote al modello. 

### Notebook di Addestramento (Google Colab)
Tutto il codice relativo alla pipeline di preprocessing, alla definizione del loop di training PyTorch, alla gestione sequenziale dei checkpoint e alla generazione dei grafici delle metriche è stato eseguito in ambiente cloud.
Il file lo puoi trovare al seguente link:


### Struttura dei File nella Repository

```text
RilevaTumore
├── file_test/
|   ├── images/
|   |   ├──  BRATS_404.nii.gz         # Scansioni MRI per testare l'app, sono presenti file sia di training sia di testing 
|   |   ├──  BRATS_408.nii.gz
|   |   ├──  BRATS_427.nii.gz
|   |   ├──  BRATS_442.nii.gz            
|   |   ├──  BRATS_485.nii.gz
|   |   ├──  BRATS_488.nii.gz
|   |   ├──  BRATS_491.nii.gz
|   |   ├──  BRATS_534.nii.gz
|   |   ├──  BRATS_607.nii.gz
|   |   └──  BRATS_619.nii.gz
|   └── labels/ 
|       ├──  BRATS_404.nii.gz         # Maschere del medico utilizzate durante l'addestramento 
|       ├──  BRATS_408.nii.gz
|       ├──  BRATS_427.nii.gz
|       └──  BRATS_442.nii.gz 
|   
├── pesi_3dunet_best_flair.pth        # Pesi del modello ottimizzato
├── app.py                            # Codice dell'applicazione Streamlit
├── requirements.txt                  # Elenco dipendenze del progetto
├── Analisi_Tecnica.pdf               # Report del progetto
└── README.md                         # File di documentazione generale
```

