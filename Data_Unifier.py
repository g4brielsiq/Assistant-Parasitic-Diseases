import os
import cv2
import pandas as pd
import shutil
import random
from sklearn.model_selection import train_test_split

def clean_and_create_dir(path):
    """Apaga a pasta se ela já existir para evitar arquivos fantasmas e cria uma nova."""
    if os.path.exists(path):
        print(f"[Limpeza] Removendo resíduos antigos em: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def process_trypanosoma(csv_path, base_images_dir, output_dir, partition, patch_size=96):
    """(Módulo Antigo) Extrai patches do Trypanosoma."""
    print(f"\n[TRYP-DB] Ignorando (Já processado) - Partição: {partition.upper()}")
    # Removi a lógica do Trypo aqui para o Colab focar só na Leishmania nesta rodada,
    # mas o código completo ficará no arquivo final para o GitHub.
    return 0 

def crop_leishmania_dataset1(leish_base_dir, output_dir, patch_size=96, test_size=0.2):
    """
    Lê o Dataset 1 da Leishmania, conserta a nomenclatura, acha o centro pelas máscaras,
    recorta 96x96 e divide automaticamente em pastas Train (80%) e Test (20%).
    """
    print(f"\n[LEISH-DB] Iniciando extração do Dataset 1 via Máscaras (PNG)...")
    
    images_dir = os.path.join(leish_base_dir, 'images')
    masks_dir = os.path.join(leish_base_dir, 'labelsv2')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[LEISH-DB] Erro: Pastas images/ ou labelsv2/ não encontradas em {leish_base_dir}")
        return 0

    # Pega todas as imagens (ignorando case)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    all_images = [f for f in os.listdir(images_dir) if f.endswith(valid_extensions)]
    
    if not all_images:
        print("[LEISH-DB] Erro: Nenhuma imagem encontrada.")
        return 0

    # Faz o particionamento estatístico (80/20)
    train_images, test_images = train_test_split(all_images, test_size=test_size, random_state=42)
    
    def process_partition(image_list, partition_name):
        target_dir = os.path.join(output_dir, partition_name, 'leish')
        clean_and_create_dir(target_dir)
        
        half_size = patch_size // 2
        count = 0
        
        for img_name in image_list:
            # Tradutor de Nomes (ex: "D (1).jpg" -> "D1-mask.png")
            base_name = img_name.replace(' ', '').replace('(', '').replace(')', '').split('.')[0]
            mask_name = f"{base_name}-mask.png"
            
            img_path = os.path.join(images_dir, img_name)
            mask_path = os.path.join(masks_dir, mask_name)
            
            if os.path.exists(mask_path):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None or mask is None: continue
                
                h, w, _ = img.shape
                _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for idx, contour in enumerate(contours):
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        
                        x_min, y_min = max(0, center_x - half_size), max(0, center_y - half_size)
                        x_max, y_max = min(w, center_x + half_size), min(h, center_y + half_size)
                        
                        patch = img[y_min:y_max, x_min:x_max]
                        
                        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                            # Salva com o RG rastreável
                            output_filename = f"leish_ds1_{idx}_{img_name}"
                            save_path = os.path.join(target_dir, output_filename)
                            cv2.imwrite(save_path, patch)
                            count += 1
        
        print(f"[LEISH-DB] Concluído! {count} patches gerados em {partition_name}/leish.")
        return count

    # Processa as duas listas
    print(f"-> Separando {len(train_images)} imagens para Treino e {len(test_images)} para Teste...")
    total_train = process_partition(train_images, 'train')
    total_test = process_partition(test_images, 'test')
    
    return total_train, total_test

if __name__ == '__main__':
    BASE_DRIVE_PATH = '/content/drive/MyDrive/Parasite'
    OUTPUT_PROCESSED_PATH = '/content/drive/MyDrive/Assistant-Parasitic-Diseases/data/processed'
    
    print("==================================================")
    print(" PIPELINE INCREMENTAL: PROCESSO DA LEISHMANIA ")
    print("==================================================")
    
    leish_ds1_dir = os.path.join(BASE_DRIVE_PATH, 'Leishmania', 'Dataset 1')
    
    try:
        t_train, t_test = crop_leishmania_dataset1(leish_ds1_dir, OUTPUT_PROCESSED_PATH)
        
        print(f"\n[SUCESSO OTIMIZADO - LEISHMANIA]")
        print(f"-> Total Treino: {t_train} patches")
        print(f"-> Total Teste: {t_test} patches")
        
    except Exception as e:
        print(f"[ERRO] Falha na execução do módulo: {e}")