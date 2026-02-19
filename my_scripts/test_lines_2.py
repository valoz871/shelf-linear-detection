import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# --- PATH ASSOLUTI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, 'my_data', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'my_data', 'output')

# --- CONFIGURAZIONE ---
FILE_CSV = os.path.join(OUTPUT_DIR, 'dati_linee.csv')
FILE_IMG = os.path.join(INPUT_DIR, '202500952000685_3_261_PROG_ROUT_10_1001_3.jpg')
BIN_SIZE = 20  # Più piccolo per precisione millimetrica

def smart_shelf_detection():
    df = pd.read_csv(FILE_CSV)
    img = cv2.imread(FILE_IMG)
    h, w = img.shape[:2]

    # 1. Filtro chirurgico sulle linee
    # Prendiamo solo linee molto orizzontali e abbastanza lunghe
    df_horiz = df[((df['Angolo'] < 5) | (df['Angolo'] > 175)) & (df['Lunghezza'] > 250)].copy()
    
    # 2. Creazione istogramma di densità
    num_bins = int(h / BIN_SIZE)
    density = np.zeros(num_bins)
    for _, row in df_horiz.iterrows():
        bin_idx = int(((row['Y1'] + row['Y2']) / 2) // BIN_SIZE)
        if 0 <= bin_idx < num_bins:
            density[bin_idx] += row['Lunghezza']

    # 3. Trovare i picchi (i ripiani reali)
    # Usiamo una soglia dinamica basata sulla media per ignorare il rumore
    threshold = np.mean(density) * 3 
    peaks = np.where(density > threshold)[0]

    # 4. Raggruppamento dei bin vicini (per non avere 10 linee sullo stesso ripiano)
    final_shelves = []
    if len(peaks) > 0:
        # Troviamo i centri dei gruppi di bin consecutivi
        temp_shelves = []
        current_group = [peaks[0]]
        for i in range(1, len(peaks)):
            if peaks[i] == peaks[i-1] + 1:
                current_group.append(peaks[i])
            else:
                temp_shelves.append(int(np.mean(current_group) * BIN_SIZE))
                current_group = [peaks[i]]
        temp_shelves.append(int(np.mean(current_group) * BIN_SIZE))

        # --- NOVITÀ: Filtro di Distanza Minima (es. 150 pixel) ---
        # Ordiniamo per densità decrescente (diamo priorità ai ripiani più "forti")
        temp_shelves.sort(key=lambda y: density[int(y//BIN_SIZE)], reverse=True)
        
        MIN_DIST = 200  # Distanza minima tra due ripiani reali in pixel
        for y in temp_shelves:
            if all(abs(y - existing_y) > MIN_DIST for existing_y in final_shelves):
                final_shelves.append(y)
        
        final_shelves.sort() # Riordiniamo dall'alto al basso per il disegno
        
    # 5. Disegno e Output
    print(f"📍 Ripiani rilevati alle coordinate Y: {final_shelves}")
    for y in final_shelves:
        cv2.line(img, (0, y), (w, y), (0, 0, 255), 4)
        cv2.putText(img, f"METALLO Y={y}", (20, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "scaffale_final_debug.jpg"), img)
    
    # Plot per capire cosa vede l'algoritmo
    plt.figure(figsize=(10,6))
    plt.plot(density)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.title("Analisi Densità Lineare (Metallo)")
    plt.savefig(os.path.join(OUTPUT_DIR, "istogramma_debug.png"))

if __name__ == "__main__":
    smart_shelf_detection()