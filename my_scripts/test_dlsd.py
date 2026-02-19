import cv2
import torch
import numpy as np
import os
import sys
import csv
from deeplsd.models.deeplsd_inference import DeepLSD

# --- PATH ASSOLUTI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(PROJECT_ROOT, 'my_data', 'input')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'my_data', 'output')
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')

# 1. Configurazione Device per Mac M4
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🚀 Device: {device}")

# 2. Definizione del modello
conf = {
    'detect_lines': True,
    'line_detection_params': {
        'merge': True,
    }
}
model = DeepLSD(conf)

# 3. Caricamento pesi
ckpt_path = os.path.join(WEIGHTS_DIR, 'deeplsd_md.tar')

if not os.path.exists(ckpt_path):
    print(f"❌ Errore: Il file {ckpt_path} non esiste!")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✅ Modello caricato con successo su M4!")

# 4. Caricamento Immagine
img_name = os.path.join(INPUT_DIR, '202500952000685_3_261_PROG_ROUT_10_1001_3.jpg')
img = cv2.imread(img_name)

if img is None:
    print(f"❌ Errore: Immagine '{img_name}' non trovata!")
    sys.exit(1)

img_draw = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
input_tensor = torch.tensor(gray[None, None] / 255., dtype=torch.float).to(device)

# 5. Inferenza
print("🧠 Elaborazione linee in corso...")
with torch.no_grad():
    out = model({'image': input_tensor})
    lines = out['lines'][0]
    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()

# 6. Elaborazione Dati, Output Testuale e Scrittura CSV
csv_file = os.path.join(OUTPUT_DIR, 'dati_linee.csv')
print(f"\n{'ID':<5} | {'X1':<7} | {'Y1':<7} | {'X2':<7} | {'Y2':<7} | {'LUNGH.':<8} | {'ANGOLO':<6}")
print("-" * 75)

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Intestazione del file CSV
    writer.writerow(['ID', 'X1', 'Y1', 'X2', 'Y2', 'Lunghezza', 'Angolo'])

    for i, line in enumerate(lines):
        # Gestione formati (2,2) o (4,)
        if line.shape == (2, 2):
            x1, y1, x2, y2 = line[0,0], line[0,1], line[1,0], line[1,1]
        elif line.shape == (4,):
            x1, y1, x2, y2 = line
        else:
            continue

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        
        # Salviamo nel CSV tutte le linee con lunghezza minima (rumore escluso)
        if length > 20:
            writer.writerow([i, round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), round(length, 2), round(angle, 2)])
            # Stampiamo a video per feedback immediato
            print(f"{i:<5} | {x1:<7.1f} | {y1:<7.1f} | {x2:<7.1f} | {y2:<7.1f} | {length:<8.1f} | {angle:<6.1f}°")

        # Disegno visivo sulla foto (solo linee lunghe e orizzontali/strutturali)
        if length > 200 and (angle < 10 or angle > 170):
            pt1 = (int(round(x1)), int(round(y1)))
            pt2 = (int(round(x2)), int(round(y2)))
            cv2.line(img_draw, pt1, pt2, (0, 255, 0), 3)

# 7. Salvataggio Finale e Chiusura
output_img = os.path.join(OUTPUT_DIR, 'output_test_scaffale.jpg')
cv2.imwrite(output_img, img_draw)

print(f"\n📊 ANALISI COMPLETATA")
print(f"📸 Foto salvata: {output_img}")
print(f"📄 Dati CSV salvati: {csv_file}")
print("🚪 Chiusura programma...")

sys.exit(0)