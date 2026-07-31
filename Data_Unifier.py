import os
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import math

# ==========================================
# 1. FUNÇÕES DE EXTRAÇÃO POR PARASITA
# ==========================================

def crop_trypanosoma(csv_path, images_dir, output_dir, patch_size=96):
    """Extrai recortes do Trypanosoma baseados em coordenadas CSV."""
    print("[Trypanosoma] Iniciando extração via CSV...")
    df = pd.read_csv(csv_path)
    os.makedirs(os.path.join(output_dir, 'trypo'), exist_ok=True)
    half_size = patch_size // 2
    count = 0
    
    for index, row in df.iterrows():
        img_name = row['filename']
        center_x, center_y = int(row['new_x']), int(row['new_y'])
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            
            x_min, y_min = max(0, center_x - half_size), max(0, center_y - half_size)
            x_max, y_max = min(w, center_x + half_size), min(h, center_y + half_size)
            patch = img[y_min:y_max, x_min:x_max]
            
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                save_path = os.path.join(output_dir, 'trypo', f"trypo_{index}_{img_name}")
                cv2.imwrite(save_path, patch)
                count += 1
    print(f"[Trypanosoma] Concluído! {count} patches gerados.")
    return count

def crop_schistosoma(images_dir, output_dir, patch_size=96):
    """Extrai recortes do Schistosoma iterando pelos arquivos XML."""
    print("[Schistosoma] Iniciando extração via XML (Pascal VOC)...")
    os.makedirs(os.path.join(output_dir, 'schisto'), exist_ok=True)
    half_size = patch_size // 2
    count = 0
    
    for filename in os.listdir(images_dir):
        if filename.endswith('.xml'):
            xml_path = os.path.join(images_dir, filename)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            img_name = root.find('filename').text
            img_path = os.path.join(images_dir, img_name)
            
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                objects = root.findall('object')
                
                for idx, obj in enumerate(objects):
                    if obj.find('name').text == 'Schistosoma':
                        bndbox = obj.find('bndbox')
                        xmin, ymin = int(bndbox.find('xmin').text), int(bndbox.find('ymin').text)
                        xmax, ymax = int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)
                        
                        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
                        x_min, y_min = max(0, center_x - half_size), max(0, center_y - half_size)
                        x_max, y_max = min(w, center_x + half_size), min(h, center_y + half_size)
                        patch = img[y_min:y_max, x_min:x_max]
                        
                        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                            save_path = os.path.join(output_dir, 'schisto', f"sh_{idx}_{img_name}")
                            cv2.imwrite(save_path, patch)
                            count += 1
    print(f"[Schistosoma] Concluído! {count} patches gerados.")
    return count

def crop_malaria(images_dir, output_dir, patch_size=96):
    """Extrai recortes da Malária lendo os arquivos de texto .xyc."""
    print("[Malária] Iniciando extração via centroid (.xyc)...")
    os.makedirs(os.path.join(output_dir, 'malaria'), exist_ok=True)
    half_size = patch_size // 2
    count = 0
    
    for filename in os.listdir(images_dir):
        if filename.endswith(('.png', '.jpg')):
            base_name = os.path.splitext(filename)[0]
            xyc_path = os.path.join(images_dir, f"{base_name}.xyc")
            img_path = os.path.join(images_dir, filename)
            
            if os.path.exists(xyc_path):
                img = cv2.imread(img_path)
                h, w, _ = img.shape
                
                with open(xyc_path, 'r') as f:
                    for idx, line in enumerate(f.readlines()):
                        coords = line.strip().split()
                        if len(coords) >= 2:
                            center_x, center_y = int(float(coords[0])), int(float(coords[1]))
                            x_min, y_min = max(0, center_x - half_size), max(0, center_y - half_size)
                            x_max, y_max = min(w, center_x + half_size), min(h, center_y + half_size)
                            patch = img[y_min:y_max, x_min:x_max]
                            
                            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                                save_path = os.path.join(output_dir, 'malaria', f"malaria_{idx}_{base_name}.jpg")
                                cv2.imwrite(save_path, patch)
                                count += 1
    print(f"[Malária] Concluído! {count} patches gerados.")
    return count

# ==========================================
# 2. CÁLCULO DE ENTROPIA (SHANNON)
# ==========================================

def calculate_entropy(positive_count, negative_count):
    """
    Calcula a Entropia de Shannon para medir o nível de desbalanceamento (imprevisibilidade)
    do dataset, conforme solicitado pela orientação do projeto.
    """
    total = positive_count + negative_count
    if total == 0:
        return 0.0
    
    p_pos = positive_count / total
    p_neg = negative_count / total
    
    entropy = 0.0
    if p_pos > 0:
        entropy -= p_pos * math.log2(p_pos)
    if p_neg > 0:
        entropy -= p_neg * math.log2(p_neg)
        
    return entropy

# ==========================================
# 3. EXECUÇÃO PRINCIPAL (MAIN)
# ==========================================

if __name__ == '__main__':
    # Estes caminhos serão usados no Google Colab. 
    # Apontam para o atalho da pasta da professora (Read-Only) e para o nosso output (Write).
    BASE_DRIVE_PATH = '/content/drive/MyDrive/Parasite'
    OUTPUT_PROCESSED_PATH = '/content/drive/MyDrive/Assistant-Parasitic-Diseases/data/processed'
    
    print("==================================================")
    print(" INICIANDO UNIFICAÇÃO DE DADOS (DATA UNIFIER) ")
    print("==================================================")
    
    # ATENÇÃO: As subpastas exatas dentro de 'Parasite' deverão ser ajustadas 
    # de acordo com o nome correto que você visualizar no Google Drive.
    trypo_csv = os.path.join(BASE_DRIVE_PATH, 'TRYP-DB', 'peerj-10-13470-s001.csv')
    trypo_img_dir = os.path.join(BASE_DRIVE_PATH, 'TRYP-DB', 'train') # Exemplo
    schisto_dir = os.path.join(BASE_DRIVE_PATH, 'SH-DB')
    malaria_dir = os.path.join(BASE_DRIVE_PATH, 'MP-IDB')
    
    # try:
    #     t_count = crop_trypanosoma(trypo_csv, trypo_img_dir, OUTPUT_PROCESSED_PATH)
    #     s_count = crop_schistosoma(schisto_dir, OUTPUT_PROCESSED_PATH)
    #     m_count = crop_malaria(malaria_dir, OUTPUT_PROCESSED_PATH)
    #     
    #     print("\n--- Relatório de Entropia (Balanceamento) ---")
    #     # O número de 'fundo' (negativos) será extraído em uma etapa posterior.
    #     # ex: entropy_trypo = calculate_entropy(t_count, negatives_count)
    # except Exception as e:
    #     print(f"Erro durante a execução: {e}")
    #     print("Verifique se os caminhos das pastas estão corretos no Google Drive.")