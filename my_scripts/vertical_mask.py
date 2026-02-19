import pandas as pd
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
BIN_SIZE = 20
WINDOW = 45  # Finestra ampia per catturare la prospettiva del ripiano

def shelf_detection_perspective_aware():
    df = pd.read_csv(FILE_CSV)
    img = cv2.imread(FILE_IMG)
    if img is None: return
    h, w = img.shape[:2]
    img_viz = img.copy()

    df['Y_mid'] = (df['Y1'] + df['Y2']) / 2
    df['X_min_raw'] = df[['X1', 'X2']].min(axis=1)
    df['X_max_raw'] = df[['X1', 'X2']].max(axis=1)

    # 1. Localizzazione bin (densità per trovare i ripiani)
    df_peaks = df[((df['Angolo'] < 5) | (df['Angolo'] > 175)) & (df['Lunghezza'] > 250)].copy()
    num_bins = int(h / BIN_SIZE)
    density = np.zeros(num_bins)
    for _, row in df_peaks.iterrows():
        bin_idx = int(row['Y_mid'] // BIN_SIZE)
        if 0 <= bin_idx < num_bins: density[bin_idx] += row['Lunghezza']
    
    threshold = np.mean(density) * 3
    peak_bins = np.where(density > threshold)[0]
    
    y_anchors = []
    if len(peak_bins) > 0:
        temp_y = []
        current_group = [peak_bins[0]]
        for i in range(1, len(peak_bins)):
            if peak_bins[i] == peak_bins[i-1] + 1: current_group.append(peak_bins[i])
            else:
                temp_y.append(int(np.mean(current_group) * BIN_SIZE))
                current_group = [peak_bins[i]]
        temp_y.append(int(np.mean(current_group) * BIN_SIZE))
        
        temp_y.sort(key=lambda y: density[int(y//BIN_SIZE)] if int(y//BIN_SIZE) < num_bins else 0, reverse=True)
        for y in temp_y:
            if all(abs(y - ey) > 150 for ey in y_anchors): y_anchors.append(y)
    y_anchors.sort()

    # 2. Analisi PROSPETTICA per ogni ripiano
    refined_shelves = []
    for y_peak in y_anchors:
        # Recupero TOTALE delle linee nel bin del ripiano
        bin_lines = df[
            (df['Y_mid'] >= y_peak - WINDOW) & (df['Y_mid'] <= y_peak + WINDOW) & 
            ((df['Angolo'] < 20) | (df['Angolo'] > 160))
        ]
        
        if not bin_lines.empty:
            # DISEGNO VERDE: Disegno integrale delle linee raw dal CSV
            for _, row in bin_lines.iterrows():
                cv2.line(img_viz, (int(row['X1']), int(row['Y1'])), 
                         (int(row['X2']), int(row['Y2'])), (0, 255, 0), 2)

            # Calcolo bordi X specifici per QUESTO ripiano
            # Bordo sinistro: minimo assoluto — i frammenti corti al bordo
            # sono pezzi reali del ripiano metallico, non rumore
            shelf_x_left = bin_lines['X_min_raw'].min()
            # Bordo destro: percentile alto (i prodotti smezzati al bordo
            # possono essere ignorati)
            shelf_x_right = np.percentile(bin_lines['X_max_raw'], 99.5)

            # Calcolo posizione/angolo pesata
            total_l = bin_lines['Lunghezza'].sum()
            avg_y = (bin_lines['Y_mid'] * bin_lines['Lunghezza']).sum() / total_l
            angs = bin_lines['Angolo'].apply(lambda a: a if a < 90 else a - 180)
            avg_a = (angs * bin_lines['Lunghezza']).sum() / total_l
            
            refined_shelves.append({
                'y': avg_y, 
                'angle': avg_a, 
                'x_left': shelf_x_left, 
                'x_right': shelf_x_right
            })

    # 3. Disegno finale con bordi adattivi
    for s in refined_shelves:
        y_c, slope = s['y'], np.tan(np.radians(s['angle']))
        x_l, x_r = s['x_left'], s['x_right']
        x_m = (x_l + x_r) / 2
        
        # Calcolo Y agli estremi usando la pendenza reale del vassoio
        y_l = int(y_c + slope * (x_l - x_m))
        y_r = int(y_c + slope * (x_r - x_m))
        
        # Disegno ripiano (Rosso) con la sua estensione reale
        cv2.line(img_draw := img_viz, (int(x_l), y_l), (int(x_r), y_r), (0, 0, 255), 6)
        
        # Disegno piccoli segmenti gialli per indicare i limiti locali del ripiano
        cv2.line(img_viz, (int(x_l), y_l - 20), (int(x_l), y_l + 20), (0, 255, 255), 3)
        cv2.line(img_viz, (int(x_r), y_r - 20), (int(x_r), y_r + 20), (0, 255, 255), 3)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "debug_prospettiva_adattiva.jpg"), img_viz)
    print(f"✅ Rilevamento completato con limiti X indipendenti per ogni ripiano.")

if __name__ == "__main__":
    shelf_detection_perspective_aware()