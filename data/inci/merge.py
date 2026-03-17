import json
import glob
import os

def merge_json_files(pattern="incibeauty_ingredients_full_*.json", output_file="inci.json"):
    # Находим все файлы, соответствующие шаблону (full_1.json, full_2.json...)
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"Файлы по шабону '{pattern}' не найдены.")
        return

    print(f"Найдено файлов для обработки: {len(files)}")
    for f in files:
        print(f" - {f}")

    merged_data = []

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Проверка: ожидаем, что внутри файла список (list)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Предупреждение: Файл {file_path} не содержит список в корне. Пропущен.")
                    
        except json.JSONDecodeError as e:
            print(f"Ошибка чтения JSON в файле {file_path}: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка при обработке {file_path}: {e}")

    # Запись результата в новый файл
    try:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # indent=2 делает файл читаемым для человека
            json.dump(merged_data, out_f, ensure_ascii=False, indent=2)
        
        print("\nГотово!")
        print(f"Всего элементов в итоговом файле: {len(merged_data)}")
        print(f"Результат сохранен в: {output_file}")
        
    except Exception as e:
        print(f"Ошибка при записи итогового файла: {e}")

if __name__ == "__main__":
    # Запуск функции
    merge_json_files()