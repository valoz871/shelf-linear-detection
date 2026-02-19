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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'debug_shelf_detector.jpg')

# Parametri rilevamento ripiani
BIN_SIZE = 20
WINDOW = 45
MIN_SHELF_DIST = 150

# Parametri bordo destro adattivo
DENSITY_SCAN_STEP = 50       # Passo di scansione X per la densità
DENSITY_DROP_THRESHOLD = 0.3  # Sotto il 30% del riferimento = gap
MAX_DY = 20.0                # ΔY massima (px) per considerare una continuazione
MAX_DANGLE = 1.5             # Δangolo massimo (°) per considerare una continuazione
GAP_CHECK_RANGE = 300        # Larghezza zona (px) da analizzare prima/dopo il gap


def find_shelf_positions(df, h):
    """
    Trova le posizioni Y dei ripiani tramite densità delle linee orizzontali.
    Restituisce lista ordinata di coordinate Y.
    """
    df_peaks = df[((df['Angolo'] < 5) | (df['Angolo'] > 175)) & (df['Lunghezza'] > 250)].copy()
    df_peaks['Y_mid'] = (df_peaks['Y1'] + df_peaks['Y2']) / 2

    num_bins = int(h / BIN_SIZE)
    density = np.zeros(num_bins)
    for _, row in df_peaks.iterrows():
        bin_idx = int(row['Y_mid'] // BIN_SIZE)
        if 0 <= bin_idx < num_bins:
            density[bin_idx] += row['Lunghezza']

    threshold = np.mean(density) * 3
    peak_bins = np.where(density > threshold)[0]

    y_anchors = []
    if len(peak_bins) > 0:
        temp_y = []
        current_group = [peak_bins[0]]
        for i in range(1, len(peak_bins)):
            if peak_bins[i] == peak_bins[i - 1] + 1:
                current_group.append(peak_bins[i])
            else:
                temp_y.append(int(np.mean(current_group) * BIN_SIZE))
                current_group = [peak_bins[i]]
        temp_y.append(int(np.mean(current_group) * BIN_SIZE))

        temp_y.sort(key=lambda y: density[int(y // BIN_SIZE)] if int(y // BIN_SIZE) < num_bins else 0,
                    reverse=True)
        for y in temp_y:
            if all(abs(y - ey) > MIN_SHELF_DIST for ey in y_anchors):
                y_anchors.append(y)

    y_anchors.sort()
    return y_anchors


def _get_zone_properties(bin_lines, x_start, x_end):
    """Calcola Y medio e angolo medio delle linee in una zona X."""
    zone = bin_lines[
        (bin_lines['X_min_raw'] < x_end) & (bin_lines['X_max_raw'] > x_start)
    ]
    if zone.empty:
        return None, None, 0

    total_l = zone['Lunghezza'].sum()
    if total_l == 0:
        return None, None, 0

    avg_y = (zone['Y_mid'] * zone['Lunghezza']).sum() / total_l
    angs = zone['Angolo'].apply(lambda a: a if a < 90 else a - 180)
    avg_a = (angs * zone['Lunghezza']).sum() / total_l
    return avg_y, avg_a, len(zone)


def find_shelf_right_edge(bin_lines, w):
    """
    Trova il bordo destro di un ripiano con logica a due fasi:

    1) Scansiona la densità lungo X per trovare i gap (dove la densità crolla
       sotto il 30% del livello di riferimento centrale).

    2) Se dopo un gap la densità risale (= ci sono altre linee), confronta
       Y medio e angolo di quelle linee con il "core" del ripiano:
       - Se ΔY < 15px E Δangolo < 1° → continuazione dello stesso ripiano, prosegui
       - Altrimenti → modulo diverso, fermati al gap
    """
    x_mins = bin_lines['X_min_raw'].values
    x_maxs = bin_lines['X_max_raw'].values
    lengths = bin_lines['Lunghezza'].values

    scan_xs = np.arange(0, w, DENSITY_SCAN_STEP)
    density = np.zeros(len(scan_xs))

    for i, xp in enumerate(scan_xs):
        mask = (x_mins < xp) & (x_maxs > xp)
        density[i] = lengths[mask].sum()

    if density.max() == 0:
        return np.percentile(bin_lines['X_max_raw'], 99.5)

    # Livello di riferimento dalla zona centrale (10%-70% dell'immagine)
    central_start = int(len(density) * 0.10)
    central_end = int(len(density) * 0.70)
    central_density = density[central_start:central_end]

    if len(central_density) == 0 or central_density.max() == 0:
        return np.percentile(bin_lines['X_max_raw'], 99.5)

    reference_level = np.percentile(central_density, 75)
    if reference_level == 0:
        return np.percentile(bin_lines['X_max_raw'], 99.5)

    drop_threshold = reference_level * DENSITY_DROP_THRESHOLD

    # Proprietà del "core" del ripiano (zona 10%-50% larghezza immagine)
    core_y, core_ang, _ = _get_zone_properties(bin_lines, w * 0.10, w * 0.50)
    if core_y is None:
        # Fallback: usa tutta la metà sinistra
        core_y, core_ang, _ = _get_zone_properties(bin_lines, 0, w * 0.50)
    if core_y is None:
        return np.percentile(bin_lines['X_max_raw'], 99.5)

    # Scansiona da sinistra a destra: traccia i gap e verifica le riprese
    above = density >= drop_threshold
    right_edge = scan_xs[-1]
    in_gap = False
    gap_start_x = 0

    # Trova l'ultimo punto sopra soglia prima di eventuali gap
    last_good_x = scan_xs[0]

    for i in range(len(scan_xs)):
        if above[i]:
            if in_gap:
                # La densità è risalita dopo un gap — verifica coerenza
                gap_end_x = scan_xs[i]
                after_y, after_ang, after_n = _get_zone_properties(
                    bin_lines, gap_end_x, gap_end_x + GAP_CHECK_RANGE
                )

                if after_y is not None and after_n >= 2:
                    dy = abs(after_y - core_y)
                    da = abs(after_ang - core_ang)

                    if dy <= MAX_DY and da <= MAX_DANGLE:
                        # Continuazione dello stesso ripiano → prosegui
                        in_gap = False
                        last_good_x = scan_xs[i]
                        continue
                    else:
                        # Modulo diverso → fermati al gap
                        right_edge = gap_start_x
                        break
                else:
                    # Poche linee dopo il gap: ignora, fermati
                    right_edge = gap_start_x
                    break
            else:
                last_good_x = scan_xs[i]
        else:
            if not in_gap and i > 0 and above[i - 1]:
                # Inizio di un gap
                in_gap = True
                gap_start_x = scan_xs[i]

    # Se siamo usciti dal loop senza break
    if in_gap:
        # Il gap arriva fino alla fine dell'immagine
        right_edge = gap_start_x
    elif not in_gap and above[-1]:
        # Tutto continuo fino alla fine — prendiamo l'ultimo punto sopra soglia + margine
        right_edge = last_good_x + DENSITY_SCAN_STEP

    return float(min(right_edge, scan_xs[-1]))


def analyze_shelves(df, y_anchors, w):
    """
    Per ogni ripiano calcola posizione, angolo e bordi X.
    Il bordo destro è calcolato individualmente per ogni ripiano
    cercando dove la densità orizzontale crolla (= fine del modulo).
    """
    df = df.copy()
    df['Y_mid'] = (df['Y1'] + df['Y2']) / 2
    df['X_min_raw'] = df[['X1', 'X2']].min(axis=1)
    df['X_max_raw'] = df[['X1', 'X2']].max(axis=1)

    refined_shelves = []
    for y_peak in y_anchors:
        bin_lines = df[
            (df['Y_mid'] >= y_peak - WINDOW) & (df['Y_mid'] <= y_peak + WINDOW) &
            ((df['Angolo'] < 20) | (df['Angolo'] > 160))
        ]

        if bin_lines.empty:
            continue

        # Bordo sinistro: minimo assoluto
        shelf_x_left = bin_lines['X_min_raw'].min()

        # Bordo destro: analisi densità (trova dove le linee di questo ripiano finiscono)
        shelf_x_right = find_shelf_right_edge(bin_lines, w)

        # Posizione e angolo pesati per lunghezza
        # Usa solo le linee dentro i bordi trovati per il calcolo
        inner_lines = bin_lines[
            (bin_lines['X_min_raw'] < shelf_x_right) &
            (bin_lines['X_max_raw'] > shelf_x_left)
        ]
        if inner_lines.empty:
            inner_lines = bin_lines

        total_l = inner_lines['Lunghezza'].sum()
        avg_y = (inner_lines['Y_mid'] * inner_lines['Lunghezza']).sum() / total_l
        angs = inner_lines['Angolo'].apply(lambda a: a if a < 90 else a - 180)
        avg_a = (angs * inner_lines['Lunghezza']).sum() / total_l

        refined_shelves.append({
            'y': avg_y,
            'angle': avg_a,
            'x_left': shelf_x_left,
            'x_right': shelf_x_right,
            'n_lines': len(bin_lines)
        })

    return refined_shelves


def draw_results(img, df, y_anchors, refined_shelves):
    """
    Disegna su img:
      - Verde: linee raw orizzontali per ogni ripiano
      - Rosso: linee dei ripiani rilevati con prospettiva
      - Giallo: marker bordi laterali
    """
    h, w = img.shape[:2]
    img_viz = img.copy()

    df = df.copy()
    df['Y_mid'] = (df['Y1'] + df['Y2']) / 2

    # Disegno linee raw verdi per ogni ripiano
    for y_peak in y_anchors:
        bin_lines = df[
            (df['Y_mid'] >= y_peak - WINDOW) & (df['Y_mid'] <= y_peak + WINDOW) &
            ((df['Angolo'] < 20) | (df['Angolo'] > 160))
        ]
        for _, row in bin_lines.iterrows():
            cv2.line(img_viz, (int(row['X1']), int(row['Y1'])),
                     (int(row['X2']), int(row['Y2'])), (0, 255, 0), 2)

    # Disegno ripiani (rosso) con prospettiva
    for s in refined_shelves:
        y_c = s['y']
        slope = np.tan(np.radians(s['angle']))
        x_l, x_r = s['x_left'], s['x_right']
        x_m = (x_l + x_r) / 2

        y_l = int(y_c + slope * (x_l - x_m))
        y_r = int(y_c + slope * (x_r - x_m))

        # Linea rossa del ripiano
        cv2.line(img_viz, (int(x_l), y_l), (int(x_r), y_r), (0, 0, 255), 6)
        # Marker gialli ai bordi
        cv2.line(img_viz, (int(x_l), y_l - 20), (int(x_l), y_l + 20), (0, 255, 255), 3)
        cv2.line(img_viz, (int(x_r), y_r - 20), (int(x_r), y_r + 20), (0, 255, 255), 3)

    return img_viz


def main():
    print("🔍 Shelf Detector — Rilevamento scaffale con bordi adattivi per ripiano")

    df = pd.read_csv(FILE_CSV)
    img = cv2.imread(FILE_IMG)
    if img is None:
        print(f"❌ Errore: Immagine non trovata: {FILE_IMG}")
        return
    h, w = img.shape[:2]
    print(f"📸 Immagine: {w}x{h}, {len(df)} linee nel CSV")

    # Fase 1: Localizzazione ripiani
    print("\n--- Fase 1: Localizzazione ripiani ---")
    y_anchors = find_shelf_positions(df, h)
    print(f"  📍 {len(y_anchors)} ripiani trovati: Y = {y_anchors}")

    # Fase 2: Analisi prospettica con bordi adattivi
    print("\n--- Fase 2: Analisi prospettica (bordi adattivi) ---")
    refined_shelves = analyze_shelves(df, y_anchors, w)
    for i, s in enumerate(refined_shelves):
        print(f"  Ripiano {i+1}: Y={s['y']:.0f}, angolo={s['angle']:.2f}°, "
              f"X=[{s['x_left']:.0f}..{s['x_right']:.0f}], {s['n_lines']} linee")

    # Fase 3: Disegno risultati
    print("\n--- Fase 3: Output ---")
    img_result = draw_results(img, df, y_anchors, refined_shelves)
    cv2.imwrite(OUTPUT_FILE, img_result)
    print(f"  ✅ Salvato: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
