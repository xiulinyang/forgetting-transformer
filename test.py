import numpy as np
import random

data = np.load("/Users/xiulinyang/Downloads/batch_0_to_1000.npy")  # don't forget .npy
num_rows = data.shape[0]
num_eval = int(num_rows * 0.03)

print(data[:3])
# rows_sample = random.sample(range(num_rows), num_eval)
#
# eval_data = data[rows_sample]
#
# np.save("eval.npy", eval_data)
#
# print(f"Eval shape: {eval_data.shape}")