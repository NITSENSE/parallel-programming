import subprocess
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # без дисплея — сохраняем только в файл
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
THREADS = [1, 2, 4, 8]          # 8 потоков на 4 ядрах (SMT)

def compile_cpp():
    src = "matrix_multiply_omp.cpp"
    out = "matrix_multiply_omp.exe" if sys.platform == "win32" else "matrix_multiply_omp"
    cmd = f"g++ -O2 -fopenmp {src} -o {out}"
    print(f"Компиляция: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Ошибка компиляции:\n", res.stderr)
        sys.exit(1)
    print("Компиляция успешна.\n")

def run_command(cmd, timeout=600):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"Таймаут ({timeout} сек): {' '.join(cmd)}")
        return -1, "", "Timeout"

def extract_time(output: str):
    for line in output.splitlines():
        if line.startswith("Время:"):
            try:
                return float(line.split()[1])
            except (IndexError, ValueError):
                pass
    return None

def main():
    os.makedirs("data", exist_ok=True)
    compile_cpp()
    exe = "matrix_multiply_omp.exe" if sys.platform == "win32" else "./matrix_multiply_omp"

    # Таблица результатов: матрица times[N][t] = время
    results = {sz: {} for sz in SIZES}

    for N in SIZES:
        # Генерация матриц (достаточно один раз на размер)
        ret, out, err = run_command([sys.executable, "generate_matrices.py", str(N)])
        if ret != 0:
            print(f"Ошибка генерации матриц для N={N}: {err}")
            continue

        for thr in THREADS:
            print(f"--- N={N}, threads={thr} ---")
            best_t = None
            for attempt in range(3):
                ret, out, err = run_command([exe, str(N), str(thr)])
                if ret != 0:
                    print(f"Ошибка запуска (попытка {attempt}): {err}")
                    continue
                t = extract_time(out)
                if t is not None:
                    if best_t is None or t < best_t:
                        best_t = t
                else:
                    print(f"Не удалось извлечь время (попытка {attempt}). Вывод:\n{out}")
            if best_t is None:
                print(f"Для N={N}, threads={thr} время не получено.\n")
                results[N][thr] = None
            else:
                results[N][thr] = best_t
                # Верификация только для одного запуска (например, первого успешного)
                ret, out, err = run_command([sys.executable, "verify_result.py", str(N)])
                if "НЕ пройдена" in out or ret != 0:
                    print("Верификация НЕ пройдена! Результат помечен как некорректный.")
                    results[N][thr] = None
                else:
                    print(f"Верификация успешна. Минимальное время: {best_t:.4f} с\n")

    # Сохраняем CSV
    with open("performance_omp_results.csv", "w") as f:
        header = "Size," + ",".join([f"T{t}_time" for t in THREADS])
        f.write(header + "\n")
        for N in SIZES:
            line = str(N)
            for t in THREADS:
                val = results[N].get(t)
                if val is not None:
                    line += f",{val:.6f}"
                else:
                    line += ",NA"
            f.write(line + "\n")

    # Построение графиков
    # 1. Время от размера для каждого числа потоков
    plt.figure(figsize=(10, 6))
    for thr in THREADS:
        times = [results[N][thr] for N in SIZES if results[N][thr] is not None]
        sizes = [N for N in SIZES if results[N][thr] is not None]
        if times:
            plt.plot(sizes, times, 'o-', label=f'{thr} поток(ов)')
    plt.xlabel('Размер матрицы N')
    plt.ylabel('Время (с)')
    plt.title('Зависимость времени выполнения от размера матрицы')
    plt.grid(True)
    plt.legend()
    plt.savefig('time_vs_size.png')
    plt.close()

    # 2. Ускорение S = T(1) / T(p)
    plt.figure(figsize=(10, 6))
    for thr in THREADS[1:]:
        speedups = []
        sizes_valid = []
        for N in SIZES:
            t1 = results[N].get(1)
            tp = results[N].get(thr)
            if t1 and tp:
                speedups.append(t1 / tp)
                sizes_valid.append(N)
        if speedups:
            plt.plot(sizes_valid, speedups, 's-', label=f'{thr} потоков')
    plt.xlabel('Размер матрицы N')
    plt.ylabel('Ускорение')
    plt.title('Ускорение относительно одного потока')
    plt.grid(True)
    plt.legend()
    plt.savefig('speedup.png')
    plt.close()

    # 3. Эффективность E = S / p
    plt.figure(figsize=(10, 6))
    for thr in THREADS[1:]:
        eff = []
        sizes_valid = []
        for N in SIZES:
            t1 = results[N].get(1)
            tp = results[N].get(thr)
            if t1 and tp:
                s = t1 / tp
                eff.append(s / thr)
                sizes_valid.append(N)
        if eff:
            plt.plot(sizes_valid, eff, 'd-', label=f'{thr} потоков')
    plt.xlabel('Размер матрицы N')
    plt.ylabel('Эффективность')
    plt.title('Эффективность распараллеливания')
    plt.grid(True)
    plt.legend()
    plt.savefig('efficiency.png')
    plt.close()

    print("Все результаты сохранены: performance_omp_results.csv, time_vs_size.png, speedup.png, efficiency.png")

if __name__ == "__main__":
    main()
