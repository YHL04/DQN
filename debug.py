import numpy as np


target = np.arange(6).reshape(2, 3).astype('float32')
q_value = np.max(target, axis=1)
print(target)
print(q_value)

y = np.arange(2) + 0.99 * q_value
print(y)
y = np.where(np.array([False, True]), np.arange(2), y)
print('y', y)
print('action', np.array([0, 1]))

print('before \n', target)
target[np.arange(2), np.array([1, 2])] = y
print('after \n', target)

