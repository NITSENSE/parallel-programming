#!/usr/bin/env python3
import os
import subprocess
import time

BASE_DIR = os.path.expanduser("~/mpi_lab")
MPI_DIR = "/soft/openmpi-1.8"

SIZES = [200, 400, 800, 1200, 1600, 2000]
PROC_CONFIGS = [
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (1, 12),
    (2, 8),
]

def get_time_limit(N):
    if N <= 1000:
        return "00:15:00"
    elif N <= 2000:
        return "00:30:00"
    else:
        return "01:00:00"

# Автоматическая установка numpy, если библиотека отсутствует в окружении пользователя
try:
    import numpy
except ImportError:
    print("Библиотека numpy не найдена. Установка...")
    subprocess.run(["pip3", "install", "--user", "numpy"], check=False)

# Создание папок для структурированного хранения файлов сценариев и результатов
RESULTS_DIR = os.path.join(BASE_DIR, "results")
JOBS_DIR = os.path.join(BASE_DIR, "jobs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

MAX_JOBS_IN_QUEUE = 15
submitted = 0

for N in SIZES:
    for nodes, ntasks in PROC_CONFIGS:
        total_procs = nodes * ntasks
        job_name = f"mpi_N{N}_P{total_procs}"
        time_req = get_time_limit(N)

        # Контроль лимита одновременно находящихся задач в очереди Slurm
        while True:
            res = subprocess.run(
                ["squeue", "-u", os.environ["USER"], "-h"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            num_jobs = len([l for l in res.stdout.strip().split('\n') if l.strip()])
            if num_jobs < MAX_JOBS_IN_QUEUE:
                break
            print(f"В очереди {num_jobs} задач (лимит {MAX_JOBS_IN_QUEUE}), ждём 30 секунд...")
            time.sleep(30)

        # Уникальная временная папка для предотвращения конфликтов при параллельной работе с файлами
        run_dir_name = f"run_N{N}_P{total_procs}"
        run_dir_path = os.path.join(BASE_DIR, run_dir_name)

        # Генерация содержимого bash-скрипта для отправки в планировщик задач Slurm
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time_req}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --output={RESULTS_DIR}/{job_name}_%j.out
#SBATCH --error={RESULTS_DIR}/{job_name}_%j.err

# Настройка путей окружения для корректной работы кастомной версии OpenMPI на узлах
export PATH={MPI_DIR}/bin:$PATH
export LD_LIBRARY_PATH={MPI_DIR}/lib:$LD_LIBRARY_PATH
export OPAL_PREFIX={MPI_DIR}

# Создание изолированной директории запуска и переход в неё
mkdir -p {run_dir_path}
cd {run_dir_path}

echo "=== Запуск: N={N}, P={total_procs} ==="
echo "Используемые узлы: $SLURM_JOB_NODELIST"

# Генерация матриц в локальной директории задачи
python3 {BASE_DIR}/generate_matrices.py {N}

# Запуск параллельного перемножения матриц нативным планировщиком srun
srun {BASE_DIR}/matrix_multiply_mpi {N}

# Верификация полученного результата с помощью numpy
python3 {BASE_DIR}/verify_result.py {N}

echo "=== Завершено: N={N}, P={total_procs} ==="

# Очистка диска от тяжелых временных текстовых файлов матриц
rm -rf {run_dir_path}/data
cd {BASE_DIR}
rmdir {run_dir_path}
"""

        script_file = os.path.join(JOBS_DIR, f"job_{job_name}.sh")
        with open(script_file, "w") as f:
            f.write(script)

        # Отправка задачи в очередь с повторной попыткой при возникновении ошибки сети/планировщика
        proc = subprocess.run(
            ["sbatch", script_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        if proc.returncode == 0:
            submitted += 1
            job_id = proc.stdout.strip().split()[-1] if proc.stdout else "???"
            print(f"[{submitted}] Отправлено: {job_name} (ID: {job_id}) (N={N}, P={total_procs}, время={time_req})")
        else:
            print(f"Ошибка отправки {job_name}: {proc.stderr.strip()}. Повторная попытка через 60 секунд...")
            time.sleep(60)
            proc = subprocess.run(
                ["sbatch", script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if proc.returncode == 0:
                submitted += 1
                job_id = proc.stdout.strip().split()[-1] if proc.stdout else "???"
                print(f"[{submitted}] Отправлено (повтор): {job_name} (ID: {job_id})")
            else:
                print(f"Не удалось отправить {job_name}, пропускаем.")

print(f"\nВсего успешно отправлено задач: {submitted}")