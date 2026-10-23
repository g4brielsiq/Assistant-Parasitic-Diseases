import os
import cv2
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def ensure_dir(path):
    """Garante que o diretório exista sem apagar dados prévios."""
    os.makedirs(path, exist_ok=True)

def reset_dir(path):
    """Apaga a pasta e recria do zero (usar apenas ao reiniciar extrações completas)."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

# ==============================================================================
# 1. MÓDULO TRYPANOSOMA (Anotação Pontual CSV -> Janela Fixa 96x96)
# ==============================================================================
def process_trypanosoma(csv_path, images_dir, output_dir, patch_size=96):
    """
    Recorta patches de 96x96 centrados nas coordenadas (x, y) do CSV original.
    Respeita a divisão oficial de 'set' ('train' e 'test') indicada no arquivo.
    """
    print("\n[TRYPANOSOMA] Iniciando extração a partir do CSV...")
    if not os.path.exists(csv_path) or not os.path.exists(images_dir):
        print(f"[TRYP] Erro: CSV ou pasta de imagens não encontrados.")
        return 0, 0

    df = pd.read_csv(csv_path)
    half_size = patch_size // 2
    counts = {'train': 0, 'test': 0}

    ensure_dir(os.path.join(output_dir, 'train', 'tryp'))
    ensure_dir(os.path.join(output_dir, 'test', 'tryp'))

    for filename, group in df.groupby('filename'):
        img_path = os.path.join(images_dir, filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape

        for idx, row in group.iterrows():
            cx, cy = int(row['x']), int(row['y'])
            split = str(row['set']).strip().lower()
            if split not in ('train', 'test'):
                split = 'train'

            x_min, y_min = max(0, cx - half_size), max(0, cy - half_size)
            x_max, y_max = min(w, cx + half_size), min(h, cy + half_size)

            patch = img[y_min:y_max, x_min:x_max]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                out_name = f"tryp_{row['structure'].lower()}_{idx}_{filename}"
                save_path = os.path.join(output_dir, split, 'tryp', out_name)
                cv2.imwrite(save_path, patch)
                counts[split] += 1

    print(f"[TRYPANOSOMA] Concluído: {counts['train']} treino | {counts['test']} teste.")
    return counts['train'], counts['test']

# ==============================================================================
# 2. MÓDULOS LEISHMANIA (DS1, DS2 e DeepLeish)
# ==============================================================================
def crop_leishmania_masks(base_dir, output_dir, ds_prefix, patch_size=96, test_size=0.2):
    """Extrai patches 96x96 de Leishmania via momentos de máscaras binárias (DS1 e DS2)."""
    print(f"\n[LEISHMANIA - {ds_prefix.upper()}] Extração via máscaras binárias...")
    images_dir = os.path.join(base_dir, 'images')
    masks_dir = os.path.join(base_dir, 'labelsv2')

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print(f"[LEISH] Diretórios images/ ou labelsv2/ ausentes em: {base_dir}")
        return 0, 0

    valid_exts = ('.jpg', '.jpeg', '.png')
    all_imgs = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    if not all_imgs:
        return 0, 0

    train_imgs, test_imgs = train_test_split(all_imgs, test_size=test_size, random_state=42)
    half_size = patch_size // 2

    def process_split(img_list, split_name):
        target_dir = os.path.join(output_dir, split_name, 'leish')
        ensure_dir(target_dir)
        count = 0

        for img_name in img_list:
            base_name = img_name.replace(' ', '').replace('(', '').replace(')', '').split('.')[0]
            mask_name = f"{base_name}-mask.png"
            mask_path = os.path.join(masks_dir, mask_name)
            img_path = os.path.join(images_dir, img_name)

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            h, w, _ = img.shape
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for idx, c in enumerate(contours):
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    x_min, y_min = max(0, cx - half_size), max(0, cy - half_size)
                    x_max, y_max = min(w, cx + half_size), min(h, cy + half_size)

                    patch = img[y_min:y_max, x_min:x_max]
                    if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                        out_name = f"leish_{ds_prefix}_{idx}_{img_name}"
                        cv2.imwrite(os.path.join(target_dir, out_name), patch)
                        count += 1
        return count

    n_train = process_split(train_imgs, 'train')
    n_test = process_split(test_imgs, 'test')
    print(f"[LEISHMANIA - {ds_prefix.upper()}] Concluído: {n_train} treino | {n_test} teste.")
    return n_train, n_test

def crop_leishmania_deepleish(deepleish_dir, output_dir, margin_pct=0.10):
    """
    Extrai amastigotas completas via anotações YOLO (.txt) com caixas delimitadoras dinâmicas.
    Mantém as partições oficiais já existentes (train/val/test).
    """
    print("\n[LEISHMANIA - DEEPLEISH] Extração dinâmica retangular (YOLO)...")
    if not os.path.exists(deepleish_dir):
        print(f"[DEEPLEISH] Pasta não encontrada: {deepleish_dir}")
        return 0, 0

    split_map = {'train': 'train', 'valid': 'train', 'test': 'test'}
    counts = {'train': 0, 'test': 0}

    for dl_split, out_split in split_map.items():
        imgs_path = os.path.join(deepleish_dir, dl_split, 'images')
        lbls_path = os.path.join(deepleish_dir, dl_split, 'labels')
        dest_dir = os.path.join(output_dir, out_split, 'leish')
        ensure_dir(dest_dir)

        if not os.path.exists(imgs_path) or not os.path.exists(lbls_path):
            continue

        for img_file in os.listdir(imgs_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            txt_file = os.path.splitext(img_file)[0] + '.txt'
            txt_path = os.path.join(lbls_path, txt_file)
            if not os.path.exists(txt_path):
                continue

            img = cv2.imread(os.path.join(imgs_path, img_file))
            if img is None:
                continue
            h_img, w_img, _ = img.shape

            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                _, x_c, y_c, w_norm, h_norm = map(float, parts[:5])
                box_w = w_norm * w_img
                box_h = h_norm * h_img

                x_min = int((x_c * w_img) - (box_w / 2) - (box_w * margin_pct))
                x_max = int((x_c * w_img) + (box_w / 2) + (box_w * margin_pct))
                y_min = int((y_c * h_img) - (box_h / 2) - (box_h * margin_pct))
                y_max = int((y_c * h_img) + (box_h / 2) + (box_h * margin_pct))

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w_img, x_max), min(h_img, y_max)

                patch = img[y_min:y_max, x_min:x_max]
                if patch.size > 0:
                    out_name = f"leish_deepleish_{dl_split}_{idx}_{img_file}"
                    cv2.imwrite(os.path.join(dest_dir, out_name), patch)
                    counts[out_split] += 1

    print(f"[LEISHMANIA - DEEPLEISH] Concluído: {counts['train']} treino | {counts['test']} teste.")
    return counts['train'], counts['test']

# ==============================================================================
# 3. MÓDULO TOXOPLASMA (Curadoria 40X e Split Estratificado)
# ==============================================================================
def process_toxoplasma(source_dir, output_dir, test_size=0.2, random_state=42):
    """Organiza patches pré-recortados de Toxoplasma gondii (40X) em train e test."""
    print("\n[TOXOPLASMA] Curadoria e divisão de patches 40X...")
    if not os.path.exists(source_dir):
        print(f"[TOXO] Pasta de origem não encontrada: {source_dir}")
        return 0, 0

    train_dest = os.path.join(output_dir, 'train', 'toxo')
    test_dest = os.path.join(output_dir, 'test', 'toxo')
    ensure_dir(train_dest)
    ensure_dir(test_dest)

    files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not files:
        return 0, 0

    train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)

    def copy_files(file_list, target_folder):
        count = 0
        for f in file_list:
            src = os.path.join(source_dir, f)
            dest = os.path.join(target_folder, f"toxo40_{f}")
            shutil.copyfile(src, dest)
            count += 1
        return count

    n_train = copy_files(train_files, train_dest)
    n_test = copy_files(test_files, test_dest)
    print(f"[TOXOPLASMA] Concluído: {n_train} treino | {n_test} teste.")
    return n_train, n_test

# ==============================================================================
# EXECUÇÃO CONTROLADA DO PIPELINE
# ==============================================================================
if __name__ == '__main__':
    BASE_DRIVE = '/content/drive/MyDrive/Parasite'
    OUTPUT_PROCESSED = '/content/drive/MyDrive/Assistant-Parasitic-Diseases/data/processed'

    print("==================================================")
    print(" PIPELINE DE PROCESSAMENTO E UNIFICAÇÃO DE DADOS ")
    print("==================================================")

    # 1. Trypanosoma
    csv_tryp = os.path.join(BASE_DRIVE, 'trypoastigote_images', 'peerj-10-13470-s001.csv')
    imgs_tryp = os.path.join(BASE_DRIVE, 'trypoastigote_images')
    # process_trypanosoma(csv_tryp, imgs_tryp, OUTPUT_PROCESSED)

    # 2. Leishmania
    leish_ds1 = os.path.join(BASE_DRIVE, 'Leishmania', 'Dataset 1')
    leish_ds2 = os.path.join(BASE_DRIVE, 'Leishmania', 'Dataset 2')
    leish_deepleish = os.path.join(BASE_DRIVE, 'Leishmania', 'DeepLeish')
    # crop_leishmania_masks(leish_ds1, OUTPUT_PROCESSED, ds_prefix='ds1')
    # crop_leishmania_masks(leish_ds2, OUTPUT_PROCESSED, ds_prefix='ds2')
    # crop_leishmania_deepleish(leish_deepleish, OUTPUT_PROCESSED)

    # 3. Toxoplasma
    toxo_src = os.path.join(BASE_DRIVE, 'Toxoplasma', 'FFGAN', 'toxo40', 'toxo40')
    # process_toxoplasma(toxo_src, OUTPUT_PROCESSED)