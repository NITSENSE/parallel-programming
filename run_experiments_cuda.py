import subprocess
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Параметры экспериментов
MATRIX_SIZES = [256, 512, 1024, 2048]  # можно добавить 4096, если хватит памяти GPU
BLOCK_SIZES = [16, 32]                 # tile size

def compile_cuda(block_size):
    """Компилирует .cu файл с заданным BLOCK_SIZE."""
    src = "matrix_multiply_cuda.cu"
    out = f"matrix_multiply_cuda_bs{block_size}"
    # Флаги: -DBLOCK_SIZE=<N>, оптимизация -O2, поддержка архитектуры (подставьте свою compute capability)
    cmd = f"nvcc -DBLOCK_SIZE={block_size} -O2 -arch=sm_61 {src} -o {out}"
    print(f"Компиляция: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Ошибка компиляции:\n", res.stderr)
        sys.exit(1)
    print("OK\n")

def run_command(cmd, timeout=600):
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"Таймаут команды: {' '.join(cmd)}")
        return -1, "", "Timeout"

def extract_time(output: str):
    """Извлекает время в секундах из вывода CUDA-программы."""
    for line in output.splitlines():
        if line.startswith("Время:"):
            try:
                return float(line.split()[1])
            except:
                pass
    return None

def main():
    os.makedirs("data", exist_ok=True)
    results = []

    for bs in BLOCK_SIZES:
        compile_cuda(bs)
        exe = f"./matrix_multiply_cuda_bs{bs}"

        for N in MATRIX_SIZES:
            print(f"--- N = {N}, BLOCK_SIZE = {bs} ---")
            # Генерация матриц
            ret, out, err = run_command([sys.executable, "generate_matrices.py", str(N)])
            if ret != 0:
                print("Ошибка генерации:", err)
                continue

            best_t = None
            for attempt in range(3):
                ret, out, err = run_command([exe, str(N)])
                if ret != 0:
                    print(f"Ошибка выполнения CUDA (попытка {attempt}):", err)
                    continue
                t = extract_time(out)
                if t and (best_t is None or t < best_t):
                    best_t = t

            if best_t is None:
                print(f"Не удалось получить время для N={N}, bs={bs}. Пропускаем.\n")
                continue

            # Верификация
            ret, out, err = run_command([sys.executable, "verify_result.py", str(N)])
            if ret != 0 or "НЕ пройдена" in out:
                print("Верификация не удалась:", out)
                continue

            print("Верификация успешна.")
            gflops = (2.0 * N**3) / best_t / 1e9
            results.append((bs, N, best_t, gflops))
            print(f"Время: {best_t:.6f} с, GFLOPS: {gflops:.2f}\n")

    if not results:
        print("Нет результатов. Завершение.")
        sys.exit(1)

    # Сохраняем таблицу в CSV
    df = pd.DataFrame(results, columns=["BlockSize", "N", "Time", "GFLOPS"])
    df.to_csv("performance_results_cuda.csv", index=False)
    print("Результаты сохранены в performance_results_cuda.csv")

    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for bs in BLOCK_SIZES:
        subset = df[df["BlockSize"] == bs]
        ax1.plot(subset["N"], subset["Time"], 'o-', label=f"BLOCK_SIZE={bs}")
        ax2.plot(subset["N"], subset["GFLOPS"], 's--', label=f"BLOCK_SIZE={bs}")

    ax1.set_xlabel("Размер матрицы N")
    ax1.set_ylabel("Время (с)")
    ax1.set_title("Зависимость времени от размера матрицы")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Размер матрицы N")
    ax2.set_ylabel("GFLOPS")
    ax2.set_title("Производительность в GFLOPS")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("performance_plot_cuda.png")
    plt.show()
    print("График сохранён в performance_plot_cuda.png")

if __name__ == "__main__":
    main()