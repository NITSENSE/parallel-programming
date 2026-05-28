#!/usr/bin/env python3
import os
import re
import glob

# Сбор и парсинг файлов результатов
results_dir = "results"
out_files = glob.glob(os.path.join(results_dir, "*.out"))

data = {} # Структура: { N: { P: (time, gflops) } }

for filepath in out_files:
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except Exception as e:
        print(f"Не удалось прочитать файл {filepath}: {e}")
        continue
        
    # Извлечение значений регулярными выражениями
    size_match = re.search(r"Размер матрицы:\s*(\d+)x\1", content)
    proc_match = re.search(r"Число процессов:\s*(\d+)", content)
    time_match = re.search(r"Время:\s*([\d\.]+)\s*секунд", content)
    gflops_match = re.search(r"Производительность:\s*([\d\.]+)\s*GFLOPS", content)
    
    if size_match and proc_match and time_match and gflops_match:
        N = int(size_match.group(1))
        P = int(proc_match.group(1))
        time_val = float(time_match.group(1))
        gflops_val = float(gflops_match.group(1))
        
        if N not in data:
            data[N] = {}
        data[N][P] = (time_val, gflops_val)

if not data:
    print("Ошибка: Данные не найдены в папке results/. Убедитесь, что задачи успешно завершились.")
    exit(1)

# Получаем уникальные и отсортированные размеры матриц и количества процессов
all_sizes = sorted(list(data.keys()))
all_procs = sorted(list({p for n in data for p in data[n]}))

# Форматированный вывод таблицы времени выполнения в формате Markdown
print("\n### Время выполнения (секунды):")
header = f"| Размерность | " + " | ".join(f"{p} проц" for p in all_procs) + " |"
separator = "| " + " | ".join(["---"] * (len(all_procs) + 1)) + " |"
print(header)
print(separator)
for N in all_sizes:
    cols = []
    for P in all_procs:
        if P in data[N]:
            cols.append(f"{data[N][P][0]:.3f}")
        else:
            cols.append("—")
    print(f"| {N:<11} | " + " | ".join(f"{col:<8}" for col in cols) + " |")

# Форматированный вывод таблицы производительности в формате Markdown
print("\n### Производительность (GFLOPS):")
print(header)
print(separator)
for N in all_sizes:
    cols = []
    for P in all_procs:
        if P in data[N]:
            cols.append(f"{data[N][P][1]:.2f}")
        else:
            cols.append("—")
    print(f"| {N:<11} | " + " | ".join(f"{col:<8}" for col in cols) + " |")