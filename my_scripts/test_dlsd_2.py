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
img_name = os.path.join(INPUT_DIR, '202500967001098_1_261_PROG_ROUT_10_1001_1.jpg')
img = cv2.imread(img_name)

if img is None:
    print(f"❌ Errore: Immagine '{img_name}' non trovata!")
    sys.exit(1)

h, w = img.shape[:2]
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

# --- 6. IDENTIFICAZIONE DINAMICA DEI LIMITI (SINISTRO E DESTRO) ---
x_starts = []
x_ends = []

for line in lines:
    if line.shape == (2, 2):
        x1, y1, x2, y2 = line[0,0], line[0,1], line[1,0], line[1,1]
    else:
        x1, y1, x2, y2 = line[0], line[1], line[2], line[3]
    
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

    if (angle < 10 or angle > 170) and length > 150:
        x_starts.append(min(x1, x2))
        x_ends.append(max(x1, x2))

if x_starts and x_ends:
    mask_x_left = np.percentile(x_starts, 10)
    mask_x_right = np.percentile(x_ends, 90)
    print(f"📐 Limiti scaffale rilevati: Sinistra X={mask_x_left:.1f}, Destra X={mask_x_right:.1f}")
    
    # Disegno linee gialle di confine
    cv2.line(img_draw, (int(mask_x_left), 0), (int(mask_x_left), h), (0, 255, 255), 3)
    cv2.line(img_draw, (int(mask_x_right), 0), (int(mask_x_right), h), (0, 255, 255), 3)
else:
    mask_x_left, mask_x_right = 0, w

# --- 7. ELABORAZIONE DATI E SCRITTURA CSV FILTRATO ---
csv_file = os.path.join(OUTPUT_DIR, 'dati_linee.csv')
print(f"\n{'ID':<5} | {'X1':<7} | {'Y1':<7} | {'X2':<7} | {'Y2':<7} | {'LUNGH.':<8} | {'ANGOLO':<6}")
print("-" * 75)

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'X1', 'Y1', 'X2', 'Y2', 'Lunghezza', 'Angolo'])

    for i, line in enumerate(lines):
        if line.shape == (2, 2):
            x1, y1, x2, y2 = line[0,0], line[0,1], line[1,0], line[1,1]
        else:
            x1, y1, x2, y2 = line[0], line[1], line[2], line[3]

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        
        # Filtro: Orizzontale e deve stare dentro il "cuore" dello scaffale
        is_horizontal = (angle < 10 or angle > 170)
        
        # Una linea è valida se attraversa o è contenuta nell'area tra i due limiti
        if length > 20 and is_horizontal:
            # Applichiamo il ritaglio (clamping) alle coordinate X
            actual_x1 = max(x1, mask_x_left)
            actual_x2 = min(x2, mask_x_right)
            
            # Se dopo il taglio la linea ha ancora senso (lunghezza positiva)
            if actual_x2 > actual_x1:
                new_length = actual_x2 - actual_x1
                writer.writerow([i, round(actual_x1, 2), round(y1, 2), round(actual_x2, 2), round(y2, 2), round(new_length, 2), round(angle, 2)])
                
                if new_length > 200:
                    print(f"{i:<5} | {actual_x1:<7.1f} | {y1:<7.1f} | {actual_x2:<7.1f} | {y2:<7.1f} | {new_length:<8.1f} | {angle:<6.1f}°")
                    # Disegno visivo in verde
                    cv2.line(img_draw, (int(actual_x1), int(y1)), (int(actual_x2), int(y2)), (0, 255, 0), 3)

# --- 8. SALVATAGGIO FINALE ---
output_img = os.path.join(OUTPUT_DIR, 'output_test_scaffale.jpg')
cv2.imwrite(output_img, img_draw)

print(f"\n📊 ANALISI COMPLETATA")
print(f"📸 Foto salvata: {output_img} (Giallo=Confini, Verde=Ripiani Filtrati)")
print(f"📄 Dati CSV filtrati salvati: {csv_file}")
sys.exit(0)