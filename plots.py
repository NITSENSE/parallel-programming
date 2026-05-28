#!/usr/bin/env python3
import matplotlib.pyplot as plt

# 1. Ваши экспериментальные данные
sizes = [200, 400, 800, 1200, 1600, 2000]
procs = [1, 2, 4, 8, 12, 16]

# Время выполнения (в секундах) для каждого числа процессов
time_data = {
    1:  [0.008, 0.058, 0.355, 1.185, 3.238, 7.947],
    2:  [0.004, 0.042, 0.176, 0.593, 1.695, 3.778],
    4:  [0.002, 0.015, 0.090, 0.393, 1.117, 1.889],
    8:  [0.001, 0.007, 0.129, 0.401, 0.946, 2.061],
    12: [0.001, 0.005, 0.054, 0.181, 0.432, 0.878],
    16: [0.001, 0.004, 0.032, 0.123, 0.281, 0.551],
}

# Производительность (в GFLOPS) для каждого числа процессов
gflops_data = {
    1:  [2.05, 2.21, 2.89, 2.92, 2.53, 2.01],
    2:  [3.76, 3.01, 5.81, 5.83, 4.83, 4.23],
    4:  [8.23, 8.68, 11.35, 8.78, 7.33, 8.47],
    8:  [17.31, 17.46, 7.93, 8.62, 8.66, 7.76],
    12: [22.82, 23.87, 19.12, 19.12, 18.94, 18.22],
    16: [23.25, 34.65, 31.53, 27.99, 29.15, 29.03],
}


# 2. ПОСТРОЕНИЕ ГРАФИКА 1: Время выполнения (Log-Log шкала)
plt.figure(figsize=(8, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

for idx, P in enumerate(procs):
    plt.loglog(sizes, time_data[P], marker='o', color=colors[idx], label=f"{P} processes")

plt.xlabel("Matrix Size (N x N)", fontsize=11)
plt.ylabel("Execution Time (seconds)", fontsize=11)
plt.title("Execution Time vs Matrix Size (Log-Log Scale)", fontsize=12, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("plot_execution_time.png", dpi=300)
plt.show()  # Покажет интерактивное окно с графиком


# 3. ПОСТРОЕНИЕ ГРАФИКОВ 2 и 3: Производительность и Эффективность (Рядом)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Левый график: Производительность (GFLOPS)
for idx, P in enumerate(procs):
    ax1.plot(sizes, gflops_data[P], marker='o', color=colors[idx], label=f"{P} proc")
ax1.set_xlabel("Matrix Size (N x N)", fontsize=11)
ax1.set_ylabel("Performance (GFLOPS)", fontsize=11)
ax1.set_title("Throughput (GFLOPS)", fontsize=12, fontweight='bold')
ax1.grid(True, ls="--", alpha=0.5)
ax1.legend(fontsize=10)

# Правый график: Эффективность параллелизации (%)
# Формула: E = T_1 / (P * T_P) * 100%
for idx, P in enumerate(procs):
    if P == 1:
        continue  # Эффективность 1 процесса всегда равна 100%, её пропускаем
    
    efficiency = []
    for i in range(len(sizes)):
        t1 = time_data[1][i]
        tp = time_data[P][i]
        eff = (t1 / (P * tp)) * 100
        efficiency.append(eff)
        
    ax2.plot(sizes, efficiency, marker='o', color=colors[idx], label=f"{P} proc")

# Линия идеальной эффективности (100%)
ax2.axhline(100, color='black', linestyle='--', alpha=0.7, label="Ideal (100%)")
ax2.set_xlabel("Matrix Size (N x N)", fontsize=11)
ax2.set_ylabel("Efficiency (%)", fontsize=11)
ax2.set_title("Parallel Efficiency", fontsize=12, fontweight='bold')
ax2.grid(True, ls="--", alpha=0.5)
ax2.legend(fontsize=10)

plt.suptitle("MPI Benchmark Analysis (Row-Major Matrix Multiply)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("plot_performance_efficiency.png", dpi=300)
plt.show()