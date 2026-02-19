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
FILE_IMG = os.path.join(INPUT_DIR, '202500967001098_1_261_PROG_ROUT_10_1001_1.jpg')
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, 'analisi_densita_ripiani.png')
OUTPUT_IMG = os.path.join(OUTPUT_DIR, 'risultato_scaffale_rilevato.jpg')

# 1. PARAMETRI DI RISOLUZIONE
BIN_SIZE = 25        

def generate_shelf_analysis():
    try:
        df = pd.read_csv(FILE_CSV)
        img = cv2.imread(FILE_IMG)
        if img is None:
            print(f"❌ Errore: Immagine {FILE_IMG} non trovata.")
            return
    except FileNotFoundError:
        print(f"❌ Errore: File non trovato.")
        return

    IMAGE_HEIGHT, IMAGE_WIDTH = img.shape[:2]

    # Calcolo coordinata Y media e filtro orizzontali (tolleranza ampia per catturare il tilt)
    df['Y_mid'] = (df['Y1'] + df['Y2']) / 2
    # Filtriamo linee che sono "ragionevolmente" orizzontali
    df_horiz = df[(df['Angolo'] < 20) | (df['Angolo'] > 160)].copy()

    # DETERMINAZIONE BINNING
    num_bins = int(np.ceil(IMAGE_HEIGHT / BIN_SIZE))
    density = np.zeros(num_bins)
    
    # Dizionario per accumulare gli angoli per ogni bin
    bin_angles = {i: [] for i in range(num_bins)}

    for _, row in df_horiz.iterrows():
        bin_idx = int(row['Y_mid'] // BIN_SIZE)
        if 0 <= bin_idx < num_bins:
            density[bin_idx] += row['Lunghezza']
            # Normalizziamo l'angolo: se > 90, lo trasformiamo nel suo supplementare negativo 
            # per avere una media coerente (es: 179° -> -1°)
            ang = row['Angolo']
            if ang > 90: ang -= 180
            bin_angles[bin_idx].append(ang)

    # --- IDENTIFICAZIONE PICCHI ---
    threshold = density.max() * 0.75
    peak_bins = np.where(density > threshold)[0]
    
    shelves_data = [] # Lista di tuple (Y_centro, Angolo_medio)
    last_peak_y = -100

    for b_idx in peak_bins:
        y_val = b_idx * BIN_SIZE
        if y_val > last_peak_y + (BIN_SIZE * 2):
            # Calcoliamo l'angolo medio delle linee in questo bin
            angles_in_bin = bin_angles[b_idx]
            avg_angle = np.mean(angles_in_bin) if angles_in_bin else 0
            shelves_data.append((y_val + BIN_SIZE/2, avg_angle))
            last_peak_y = y_val

    # --- DISEGNO SULLA FOTO CON PENDENZA REALE ---
    print(f"\n✍️ Disegno ripiani con inclinazione reale...")
    for y_center, angle in shelves_data:
        # Calcoliamo i punti estremi della linea usando la pendenza (tangente dell'angolo)
        # y = y_center + tan(angolo) * (x - x_center)
        angle_rad = np.radians(angle)
        slope = np.tan(angle_rad)
        
        # Punto X=0 (sinistra)
        x_left = 0
        y_left = int(y_center - (IMAGE_WIDTH / 2) * slope)
        
        # Punto X=IMAGE_WIDTH (destra)
        x_right = IMAGE_WIDTH
        y_right = int(y_center + (IMAGE_WIDTH / 2) * slope)

        # Disegno linea
        cv2.line(img, (x_left, y_left), (x_right, y_right), (0, 0, 255), 5)
        cv2.putText(img, f"Y={int(y_center)} Ang:{angle:.1f}", (50, int(y_center) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imwrite(OUTPUT_IMG, img)
    print(f"✅ Foto salvata con tilt corretto: {OUTPUT_IMG}")

    # --- PLOTTING (Invariato) ---
    plt.figure(figsize=(10, 8))
    plt.barh(np.arange(num_bins)*BIN_SIZE, density, height=BIN_SIZE, color='green', alpha=0.6, align='edge')
    plt.gca().invert_yaxis()
    plt.title('Profilo Densità (con correzione Angolo)')
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    plt.show()

if __name__ == "__main__":
    generate_shelf_analysis()