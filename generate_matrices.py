import sys
import os
import numpy as np

def generate_square_matrices(size: int):
    os.makedirs("data", exist_ok=True)
    A = np.random.uniform(1, 10, (size, size)).astype(np.float32)
    B = np.random.uniform(1, 10, (size, size)).astype(np.float32)
    np.savetxt("data/A.txt", A, fmt="%.6f")
    np.savetxt("data/B.txt", B, fmt="%.6f")
    print(f"Сгенерированы матрицы {size}x{size} (float32) и сохранены в data/A.txt, data/B.txt")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Использование: python generate_matrices.py <размер>")
        sys.exit(1)
    n = int(sys.argv[1])
    generate_square_matrices(n)