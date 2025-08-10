import numpy as np
import random

data = np.load("data/train.npy")  # don't forget .npy
num_rows = data.shape[0]
num_eval = int(num_rows * 0.03)

rows_sample = random.sample(range(num_rows), num_eval)

eval_data = data[rows_sample]

np.save("data/eval.npy", eval_data)

print(f"Eval shape: {eval_data.shape}")
