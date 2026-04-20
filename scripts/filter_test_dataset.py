"""
scripts/filter_test_dataset.py

Фильтрация тестовой выборки — оставляет только пары с реальными изменениями.
"""

import numpy as np
import os
import cv2
import shutil
import argparse
from tqdm import tqdm


def filter_test_dataset(data_path: str, output_path: str = None, 
                        min_change_pixels: int = 100):
    """
    Фильтрация тестовой выборки.
    
    Оставляет только те пары, где маска изменений содержит
    хотя бы min_change_pixels пикселей со значением 255.
    
    Параметры:
        data_path: путь к исходным данным (LEVIR-CD-256)
        output_path: путь для сохранения отфильтрованных данных
        min_change_pixels: минимальное количество пикселей изменений
    """
    if output_path is None:
        output_path = data_path + "_filtered"
    
    print("=" * 60)
    print("ФИЛЬТРАЦИЯ ТЕСТОВОЙ ВЫБОРКИ")
    print("=" * 60)
    print(f"Исходные данные: {data_path}")
    print(f"Выходные данные: {output_path}")
    print(f"Минимум пикселей изменений: {min_change_pixels}")
    print("=" * 60)
    
    # Создание структуры папок
    for split in ['train', 'val', 'test']:
        for subdir in ['A', 'B', 'label']:
            os.makedirs(os.path.join(output_path, split, subdir), exist_ok=True)
    
    stats = {}
    
    for split in ['train', 'val', 'test']:
        print(f"\nОбработка раздела: {split}")
        
        label_dir = os.path.join(data_path, split, "label")
        if not os.path.exists(label_dir):
            print(f"  Директория {label_dir} не найдена, пропускаем")
            continue
        
        a_dir = os.path.join(data_path, split, "A")
        b_dir = os.path.join(data_path, split, "B")
        
        # Получение списка файлов
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
        
        kept = 0
        removed = 0
        
        for label_file in tqdm(label_files, desc=f"  {split}"):
            # Загрузка маски
            label_path = os.path.join(label_dir, label_file)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            if label is None:
                print(f"    Предупреждение: не удалось загрузить {label_file}")
                removed += 1
                continue
            
            # Подсчёт пикселей изменений
            change_pixels = np.sum(label > 127)
            
            # Решение: оставить или удалить
            if change_pixels >= min_change_pixels:
                # Копируем A, B, label
                a_path = os.path.join(a_dir, label_file)
                b_path = os.path.join(b_dir, label_file)
                
                if os.path.exists(a_path) and os.path.exists(b_path):
                    shutil.copy2(a_path, os.path.join(output_path, split, "A", label_file))
                    shutil.copy2(b_path, os.path.join(output_path, split, "B", label_file))
                    shutil.copy2(label_path, os.path.join(output_path, split, "label", label_file))
                    kept += 1
                else:
                    removed += 1
            else:
                removed += 1
        
        stats[split] = {'kept': kept, 'removed': removed}
        print(f"  {split}: оставлено {kept}, удалено {removed}")
    
    # Итоговая статистика
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    
    total_kept = sum(s['kept'] for s in stats.values())
    total_removed = sum(s['removed'] for s in stats.values())
    
    print(f"Всего оставлено пар: {total_kept}")
    print(f"Всего удалено пар: {total_removed}")
    print(f"\nОтфильтрованные данные сохранены в: {output_path}")
    
    return stats


def analyze_test_dataset(data_path: str):
    """
    Анализ тестовой выборки: сколько пар с изменениями.
    """
    print("\n" + "=" * 60)
    print("АНАЛИЗ ТЕСТОВОЙ ВЫБОРКИ")
    print("=" * 60)
    
    label_dir = os.path.join(data_path, "test", "label")
    if not os.path.exists(label_dir):
        print(f"Директория {label_dir} не найдена")
        return
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.png')]
    
    change_counts = []
    empty_count = 0
    
    for label_file in tqdm(label_files, desc="Анализ"):
        label_path = os.path.join(label_dir, label_file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        if label is None:
            continue
        
        change_pixels = np.sum(label > 127)
        change_counts.append(change_pixels)
        
        if change_pixels == 0:
            empty_count += 1
    
    print(f"\nВсего файлов: {len(label_files)}")
    print(f"Пустых масок (0 изменений): {empty_count} ({100*empty_count/len(label_files):.1f}%)")
    print(f"Масок с изменениями: {len(label_files) - empty_count}")
    print(f"\nСтатистика по пикселям изменений:")
    print(f"  Минимум: {min(change_counts)}")
    print(f"  Максимум: {max(change_counts)}")
    print(f"  Среднее: {np.mean(change_counts):.1f}")
    print(f"  Медиана: {np.median(change_counts):.1f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Фильтрация тестовой выборки LEVIR-CD')
    parser.add_argument('--data_path', type=str, default='./data/LEVIR-CD-256',
                        help='Путь к данным')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Путь для сохранения отфильтрованных данных')
    parser.add_argument('--min_pixels', type=int, default=100,
                        help='Минимальное количество пикселей изменений')
    parser.add_argument('--analyze_only', action='store_true',
                        help='Только анализ, без фильтрации')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_test_dataset(args.data_path)
    else:
        filter_test_dataset(args.data_path, args.output_path, args.min_pixels)