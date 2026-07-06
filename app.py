import os
import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go  # utilizzato per il rendering 3D del cervello
import matplotlib.pyplot as plt  
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, LoadImageD, EnsureChannelFirstD, SpacingD,
    NormalizeIntensityD, CropForegroundD, DivisiblePadD, Lambdad,
    MapLabelValueD  # serve per la binarizzazione corretta delle etichette del medico
)

# Configurazione della pagina 
st.set_page_config(page_title="Rileva Tumore", layout="wide")
st.title("Segmentazione dei tumori cerebrali")
st.write(
    "La piattaforma nasce per offrire una panoramica del comportamento dell'algoritmo, "
    "al suo interno sono presenti sia i dati utilizzati in fase di training "
    "sia i dati di testing.\n\n" 
    "In questo modo, l'interfaccia permette di confrontare "
    "direttamente come il modello si comporta con i casi che ha già visto durante lo studio e con quelli che non ha mai incontrato prima."
)


# Definizione dei percorsi e dei dati
DATA_DIR = "file_test"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")


# Caricamento del modello 
@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    if os.path.exists("pesi_3dunet_best_flair.pth"):
        model.load_state_dict(torch.load("pesi_3dunet_best_flair.pth", map_location=device))
        model.eval()
        return model
    else:
        st.error(" Il file 'pesi_3dunet_best_flair.pth' non è stato trovato nella cartella principale del progetto.")
        st.stop()

model = load_model()


# Lettura automatica dei dati 
st.sidebar.header("📁 Archivio dei Pazienti")

if os.path.exists(IMAGES_DIR):

    # Elenca tutti i file di risonanza presenti nella cartella images

    lista_pazienti = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.nii', '.nii.gz'))]
    lista_pazienti.sort()
    
    if lista_pazienti:
        paziente_scelto = st.sidebar.selectbox("Seleziona il Paziente da analizzare:", lista_pazienti)
        
        percorso_immagine = os.path.join(IMAGES_DIR, paziente_scelto)
        percorso_label = os.path.join(LABELS_DIR, paziente_scelto)
        
        # Avviene una controllo automatico per verifica se per il paziente selezionato esiste la maschera del medico
        has_mask = os.path.exists(percorso_label)
        if has_mask:
            st.sidebar.success("Maschera del Medico trovata in archivio")
        else:
            st.sidebar.warning("Paziente ignoto. L'analisi sarà basata solo sulla predizione del modello.")
    else:
        st.sidebar.error(f"Nessun file trovato in {IMAGES_DIR}")
        st.stop()
else:
    st.sidebar.error(f"Cartella non trovata: {IMAGES_DIR}.")
    st.stop()


# Gestione dello stato della sessione per evitare che lo slider si resetti quando l'utente cambia paziente o ricarica la pagina
if "ultimo_paziente" not in st.session_state:
    st.session_state["ultimo_paziente"] = None


# Se l'utente cambia paziente dal menu a tendina, resetta lo stato dell'analisi
if paziente_scelto != st.session_state["ultimo_paziente"]:
    st.session_state["elaborato"] = False
    st.session_state["ultimo_paziente"] = paziente_scelto

pulsante_avvia = st.sidebar.button("🚀 AVVIA ANALISI")



# Preprocessing, inferenza e visualizzazione dei risultati
if pulsante_avvia or st.session_state.get("elaborato", False):
    
    # Esegue il preprocessing e la U-Net solo se non è già stato fatto per il paziente corrente
    if not st.session_state.get("elaborato", False):
        with st.spinner("Caricamento in corso..."):
            
            if has_mask:
                data_dict = {"image": percorso_immagine, "label": percorso_label}
                pipeline_esame = Compose([
                    LoadImageD(keys=["image", "label"]),
                    EnsureChannelFirstD(keys=["image", "label"]),
                    Lambdad(keys="image", func=lambda x: x[0:1, :, :, :] if len(x.shape) == 4 else x),
                    Lambdad(keys="label", func=lambda x: x[0:1, :, :, :] if len(x.shape) == 4 else x),
                    SpacingD(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear", "nearest"]),
                    NormalizeIntensityD(keys="image", nonzero=True),
                    CropForegroundD(keys=["image", "label"], source_key="image"),


                    # Correzione del doppio spessore: binarizza i valori 1, 2, 3 della maschera reale in 1 
                    MapLabelValueD(keys="label", orig_labels=[1, 2, 3], target_labels=[1, 1, 1]),
                    DivisiblePadD(keys=["image", "label"], k=16)
                ])
            else:
                data_dict = {"image": percorso_immagine}
                pipeline_esame = Compose([
                    LoadImageD(keys=["image"]),
                    EnsureChannelFirstD(keys=["image"]),
                    Lambdad(keys="image", func=lambda x: x[0:1, :, :, :] if len(x.shape) == 4 else x),
                    SpacingD(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
                    NormalizeIntensityD(keys="image", nonzero=True),
                    CropForegroundD(keys=["image"], source_key="image"),
                    DivisiblePadD(keys=["image"], k=16)
                ])
            
            dati_pronti = pipeline_esame(data_dict)
            input_tensor = dati_pronti["image"].unsqueeze(0).to(torch.device("cpu"))
            
            with torch.no_grad():
                output = model(input_tensor)
                pred = (torch.sigmoid(output) > 0.5).float()
                
                img_np = dati_pronti["image"][0].cpu().numpy()
                pred_np = pred[0, 0].cpu().numpy()
                mask_np = dati_pronti["label"][0].cpu().numpy() if has_mask else np.zeros_like(img_np)


            # Salva i vettori nello stato della sessione per non ricalcolarli muovendo lo slider
            st.session_state["img_np"] = img_np
            st.session_state["pred_np"] = pred_np
            st.session_state["mask_np"] = mask_np
            st.session_state["has_mask"] = has_mask
            st.session_state["elaborato"] = True


    # Recupero dei dati stabili dalla sessione
    img_np = st.session_state["img_np"]
    pred_np = st.session_state["pred_np"]
    mask_np = st.session_state["mask_np"]
    has_mask = st.session_state["has_mask"]
    
    # Calcolo delle metriche dell'IA
    volume_voxel = np.sum(pred_np > 0)
    volume_cm3 = volume_voxel / 1000.0

   
    # SEZIONI 2D 
    st.subheader(f"🔬 Ispezione Sezioni 2D - {paziente_scelto}")
    st.write("Scorri lo slider per confrontare diverse sezioni della risonanza")
    
    profondita_z = img_np.shape[2]
    fetta_selezionata = st.slider(
        "Seleziona 'fetta' (Asse Z):", 
        min_value=0, max_value=profondita_z - 1, value=profondita_z // 2
    )
    
    # Layout colonne dinamico: 3 colonne se c'è la maschera reale, altrimenti 2
    if has_mask:
        c1, c2, c3 = st.columns(3)
    else:
        c1, c3 = st.columns(2)
        c2 = None
    
    # Pannello 1: Immagine Originale
    with c1:
        st.caption("### Risonanza Originale ")
        fig_mri, ax_mri = plt.subplots(figsize=(5, 5), layout="tight")
        ax_mri.imshow(img_np[:, :, fetta_selezionata], cmap="gray")
        ax_mri.axis("off")
        st.pyplot(fig_mri, use_container_width=True)
        plt.close(fig_mri)
        
    # Pannello 2: MOstra la maschera del medico se presente in archivio
    if c2 is not None and has_mask:
        with c2:
            st.caption("### Maschera del Medico")
            fig_gt, ax_gt = plt.subplots(figsize=(5, 5), layout="tight")
            ax_gt.imshow(img_np[:, :, fetta_selezionata], cmap="gray")
            if np.sum(mask_np[:, :, fetta_selezionata] > 0) > 0:
                ax_gt.imshow(mask_np[:, :, fetta_selezionata] > 0, cmap="Reds", alpha=0.5)
            ax_gt.axis("off")
            st.pyplot(fig_gt, use_container_width=True)
            plt.close(fig_gt)

    # Pannello 3: Predizione del modello
    with c3:
        st.caption("### Predizione del Modello")
        fig_pred, ax_pred = plt.subplots(figsize=(5, 5), layout="tight")
        ax_pred.imshow(img_np[:, :, fetta_selezionata], cmap="gray")
        if np.sum(pred_np[:, :, fetta_selezionata]) > 0:
            ax_pred.imshow(pred_np[:, :, fetta_selezionata], cmap="Blues", alpha=0.5)
        ax_pred.axis("off")
        st.pyplot(fig_pred, use_container_width=True)
        plt.close(fig_pred)

    st.markdown("---")

    
    # RENDERING 3D 
    st.subheader("🧠 Rendering 3D del Cervello")
    st.write("Ruota il cervello con il mouse per esaminare la profondità tridimensionale della massa.")
    
    col_grafico, col_metriche = st.columns([3, 1])
    
    with col_grafico:
        passo = 2
        X, Y, Z = np.mgrid[0:img_np.shape[0]:passo, 0:img_np.shape[1]:passo, 0:img_np.shape[2]:passo]
        sotto_img = img_np[::passo, ::passo, ::passo]
        sotto_pred = pred_np[::passo, ::passo, ::passo]
        sotto_mask = mask_np[::passo, ::passo, ::passo] if has_mask else None

        fig_3d = go.Figure()

        # Correzione del cervello tagliato: stabilisce una soglia minima fissa vicina allo zero dello Z-score
        voxel_cervello = sotto_img[sotto_img > 0.0]
        isomin_cervello = float(np.percentile(voxel_cervello, 5)) if voxel_cervello.size > 0 else 0.1
        isomin_cervello = min(isomin_cervello, 0.12)  # Impedisce al tumore molto acceso di spegnere il cervello sano
        isomax_cervello = float(np.max(sotto_img))

        # Sezione del cervello trasparente
        fig_3d.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=sotto_img.flatten(),
            isomin=isomin_cervello, isomax=isomax_cervello,
            opacity=0.07, surface_count=3,
            colorscale="Gray", showscale=False,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))

        # Maschera Medico (Rosso) - Presente solo se l'archivio lo prevede
        if has_mask and np.sum(sotto_mask > 0) > 0:
            fig_3d.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=sotto_mask.flatten(),
                isomin=0.5, isomax=1.0,
                opacity=0.45, colorscale="Reds", showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            ))

        # Predizione dell'IA (Blu)
        if np.sum(sotto_pred) > 0:
            fig_3d.add_trace(go.Isosurface(
                x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
                value=sotto_pred.flatten(),
                isomin=0.5, isomax=1.0,
                opacity=0.60, colorscale="Blues", showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False)
            ))

        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                bgcolor="#616161" 
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            autosize=True,
        )
        st.plotly_chart(fig_3d, use_container_width=True, config={'displayModeBar': False})

    with col_metriche:
        st.markdown("### Dati Clinici")
        st.metric("Volume AI", f"{volume_cm3:.2f} cm³")
        st.metric("Conteggio totale", f"{int(volume_voxel)} voxel")
        
        if has_mask:
            # Calcolo matematico del Dice Score
            volume_medico = np.sum(mask_np > 0)
            intersezione = np.sum((pred_np > 0) * (mask_np > 0))
            dice_score = (2.0 * intersezione) / (np.sum(pred_np > 0) + np.sum(mask_np > 0) + 1e-6)
            
            st.metric("Volume Medico", f"{volume_medico / 1000.0:.2f} cm³")
            st.metric("Dice Coefficient", f"{dice_score:.4f}")

        st.info(
            "Il nucleo bianco all'interno del cervello evidenzia la lesione tumorale individuata."
        )
else:
    st.info("⬅️ Seleziona un paziente dall'archivio nella barra laterale e premi 'Avvia Analisi'.")

# eseguire nel cmd streamlit run app.py
