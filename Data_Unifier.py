import os
import cv2
import pandas as pd
import shutil

def clean_and_create_dir(path):
    """Apaga a pasta se ela já existir para evitar arquivos fantasmas e cria uma nova."""
    if os.path.exists(path):
        print(f"[Limpeza] Removendo resíduos antigos em: {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def crop_trypanosoma_partition(csv_path, base_images_dir, output_dir, partition='train', patch_size=96):
    """Extrai patches do Trypanosoma filtrando pela partição correta (train/test)."""
    print(f"\n[TRYP-DB] Iniciando extração cirúrgica para a partição: {partition.upper()}...")
    df = pd.read_csv(csv_path)
    
    # Filtra o CSV para processar apenas as linhas da partição atual
    df_filtered = df[df['set'].str.lower() == partition.lower()]
    
    # Pasta final dedicada à partição
    target_dir = os.path.join(output_dir, partition, 'trypo')
    clean_and_create_dir(target_dir)
    
    half_size = patch_size // 2
    count = 0
    
    # O diretório real da imagem depende se ela é train ou test
    images_dir = os.path.join(base_images_dir, partition)
    
    for index, row in df_filtered.iterrows():
        img_name = row['filename']
        img_path = os.path.join(images_dir, img_name)
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is None: continue
                
            h, w, _ = img.shape
            center_x, center_y = int(row['x']), int(row['y'])
            
            x_min, y_min = max(0, center_x - half_size), max(0, center_y - half_size)
            x_max, y_max = min(w, center_x + half_size), min(h, center_y + half_size)
            
            patch = img[y_min:y_max, x_min:x_max]
            
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                structure_type = str(row['structure']).lower()
                output_filename = f"trypo_{structure_type}_{index}_{img_name}"
                save_path = os.path.join(target_dir, output_filename)
                
                cv2.imwrite(save_path, patch)
                count += 1
                
    print(f"[TRYP-DB] Concluído! {count} patches gerados em {partition}/trypo.")
    return count

if __name__ == '__main__':
    BASE_DRIVE_PATH = '/content/drive/MyDrive/Parasite'
    OUTPUT_PROCESSED_PATH = '/content/drive/MyDrive/Assistant-Parasitic-Diseases/data/processed'
    
    print("==================================================")
    print(" PIPELINE INCREMENTAL: PROCESSO DO TRYPANOSOMA ")
    print("==================================================")
    
    trypo_csv = os.path.join(BASE_DRIVE_PATH, 'trypoastigote_images', 'peerj-10-13470-s001.csv')
    trypo_base_dir = os.path.join(BASE_DRIVE_PATH, 'trypoastigote_images')
    
    try:
        # Roda a extração organizada para as duas partições necessárias
        total_train = crop_trypanosoma_partition(trypo_csv, trypo_base_dir, OUTPUT_PROCESSED_PATH, partition='train')
        total_test = crop_trypanosoma_partition(trypo_csv, trypo_base_dir, OUTPUT_PROCESSED_PATH, partition='test')
        
        print(f"\n[SUCESSO OTIMIZADO]")
        print(f"-> Total Treino: {total_train} patches")
        print(f"-> Total Teste: {total_test} patches")
        print(f"-> Base TRYP-DB 100% Concluída e Estruturada!")
    except Exception as e:
        print(f"[ERRO] Falha na execução do módulo: {e}")