import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Размеры матриц для тестирования (можно изменить)
SIZES = [200, 500, 1000, 1500, 2000]  # Убрал 3000 для быстроты демонстрации

def compile_cpp():
    """Компиляция C++ программы с оптимизацией."""
    src = "matrix_multiply.cpp"
    out = "matrix_multiply.exe" if sys.platform == "win32" else "matrix_multiply"
    cmd = f"g++ -O2 {src} -o {out}"
    print(f"Компиляция: {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Ошибка компиляции:\n", res.stderr)
        sys.exit(1)
    print("Компиляция успешна.\n")

def run_command(cmd, timeout=300):
    """Выполнение команды с замером времени."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        print(f"Команда превысила таймаут ({timeout} сек): {' '.join(cmd)}")
        return -1, "", "Timeout"

def extract_time(output: str):
    """Извлекает время выполнения из вывода C++ программы."""
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

    exe = "matrix_multiply.exe" if sys.platform == "win32" else "./matrix_multiply"

    times = []
    gflops_vals = []
    valid_sizes = []   # будем сохранять только те размеры, для которых всё прошло успешно

    for N in SIZES:
        print(f"--- N = {N} ---")
        # Генерация матриц
        ret, out, err = run_command([sys.executable, "generate_matrices.py", str(N)])
        if ret != 0:
            print("Ошибка генерации матриц:", err)
            continue   # пропускаем этот размер, в CSV он не попадёт

        # Три запуска, сохраняем минимальное время
        best_t = None
        for attempt in range(1, 4):
            ret, out, err = run_command([exe, str(N)])
            if ret != 0:
                print(f"Ошибка выполнения C++ (попытка {attempt}):", err)
                continue
            t = extract_time(out)
            if t is not None:
                if best_t is None or t < best_t:
                    best_t = t
            else:
                print(f"Не удалось извлечь время из вывода (попытка {attempt}). Вывод:\n{out}")

        if best_t is None:
            print(f"Для N={N} не удалось получить корректное время. Пропускаем в итоговой таблице.\n")
            continue

        # Верификация
        ret, out, err = run_command([sys.executable, "verify_result.py", str(N)])
        if "НЕ пройдена" in out or ret != 0:
            print("Верификация не удалась:", out)
            print("Результат не будет учтён в итоговых данных.\n")
            continue

        print("Верификация успешна.")
        gflops = (2.0 * N**3) / best_t / 1e9
        times.append(best_t)
        gflops_vals.append(gflops)
        valid_sizes.append(N)
        print(f"Минимальное время: {best_t:.4f} с, GFLOPS: {gflops:.2f}\n")

    if not valid_sizes:
        print("Не удалось получить результаты ни для одного размера. Завершение.")
        sys.exit(1)

    # Сохранение CSV только с успешными размерами
    with open("performance_results.csv", "w") as f:
        f.write("Size,Time(s),GFLOPS\n")
        for i, N in enumerate(valid_sizes):
            f.write(f"{N},{times[i]:.6f},{gflops_vals[i]:.2f}\n")
    print("Результаты сохранены в performance_results.csv")

    # Графики
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(valid_sizes, times, 'o-')
    ax1.set_xlabel("Размер матрицы N")
    ax1.set_ylabel("Время (с)")
    ax1.set_title("Зависимость времени от размера")
    ax1.grid(True)

    ax2.plot(valid_sizes, gflops_vals, 's-', color='red')
    ax2.set_xlabel("Размер матрицы N")
    ax2.set_ylabel("GFLOPS")
    ax2.set_title("Производительность")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("performance_plot.png")
    plt.show()
    print("График сохранён в performance_plot.png")

if __name__ == "__main__":
    main()