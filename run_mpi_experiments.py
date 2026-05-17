import subprocess
import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROCS = [1, 2, 4, 8]

def find_mpi_executable(names):
    """Ищет исполняемый файл MPI по списку возможных имён."""
    # Стандартный поиск в PATH
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    # Дополнительные пути, специфичные для Fedora/RHEL
    extra_dirs = ['/usr/lib64/openmpi/bin', '/usr/lib/openmpi/bin']
    for d in extra_dirs:
        for name in names:
            full = os.path.join(d, name)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return None

def compile_mpi():
    compiler = find_mpi_executable(['mpicxx', 'mpic++', 'mpicxx.openmpi', 'mpic++.openmpi'])
    if not compiler:
        print("Ошибка: MPI-компилятор (mpicxx/mpic++) не найден.")
        print("Установите OpenMPI и убедитесь, что путь /usr/lib64/openmpi/bin доступен.")
        sys.exit(1)

    src = "matrix_multiply_mpi.cpp"
    out = "matrix_multiply_mpi"
    cmd = f"{compiler} -O2 {src} -o {out}"
    print(f"Компиляция ({compiler}): {cmd}")
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print("Ошибка компиляции:\n", res.stderr)
        sys.exit(1)
    print("Компиляция успешна.\n")

def run_command(cmd, timeout=600):
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
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
    compile_mpi()

    # Ищем mpirun (также с учётом нестандартных путей)
    mpirun_path = find_mpi_executable(['mpirun', 'mpirun.openmpi'])
    if not mpirun_path:
        print("Ошибка: mpirun не найден. Проверьте установку OpenMPI.")
        sys.exit(1)

    exe = "./matrix_multiply_mpi"

    times = {n: {} for n in SIZES}

    for N in SIZES:
        ret, _, err = run_command(f"{sys.executable} generate_matrices.py {N}")
        if ret != 0:
            print(f"Ошибка генерации N={N}: {err}")
            continue

        for P in PROCS:
            print(f"--- N={N}, P={P} ---")
            best_t = None
            for attempt in range(1, 4):
                cmd = f"{mpirun_path} --oversubscribe -np {P} {exe} {N}"
                ret, out, err = run_command(cmd)
                if ret != 0:
                    print(f"Ошибка MPI-запуска (попытка {attempt}): {err}")
                    continue
                t = extract_time(out)
                if t is not None:
                    if best_t is None or t < best_t:
                        best_t = t
                else:
                    print(f"Не удалось извлечь время (попытка {attempt}). Вывод:\n{out}")
            if best_t is None:
                print(f"Не получено время для N={N}, P={P}")
                continue

            ret, out, _ = run_command(f"{sys.executable} verify_result.py {N}")
            if "НЕ пройдена" in out or ret != 0:
                print(f"Верификация не пройдена: {out}")
                continue

            times[N][P] = best_t
            gflops = (2.0 * N**3) / best_t / 1e9
            print(f"Минимальное время: {best_t:.4f} с, GFLOPS: {gflops:.2f}\n")

    if not any(times[n] for n in SIZES):
        print("Нет успешных результатов. Выход.")
        sys.exit(1)

    with open("mpi_performance.csv", "w") as f:
        header = "Size," + ",".join(f"P{p}" for p in PROCS)
        f.write(header + "\n")
        for N in SIZES:
            line = str(N)
            for P in PROCS:
                t = times[N].get(P, "")
                line += f",{t:.6f}" if t != "" else ","
            f.write(line + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for N in SIZES:
        x, y = [], []
        for P in PROCS:
            if P in times[N]:
                x.append(P)
                y.append(times[N][P])
        ax1.plot(x, y, marker='o', label=f'N={N}')
    ax1.set_xlabel("Число процессов")
    ax1.set_ylabel("Время (с)")
    ax1.set_title("Время выполнения")
    ax1.legend()
    ax1.grid(True)

    for N in SIZES:
        if 1 not in times[N]:
            continue
        t1 = times[N][1]
        x, speedup = [], []
        for P in PROCS:
            if P in times[N]:
                x.append(P)
                speedup.append(t1 / times[N][P])
        ax2.plot(x, speedup, marker='s', label=f'N={N}')
    ax2.set_xlabel("Число процессов")
    ax2.set_ylabel("Ускорение")
    ax2.set_title("Ускорение (T₁ / Tₚ)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("mpi_performance.png")
    plt.show()
    print("Результаты сохранены в mpi_performance.csv и mpi_performance.png")

if __name__ == "__main__":
    main()