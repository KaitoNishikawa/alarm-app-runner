import numpy as np
x = np.load('outputs/features/893_count_feature.npy')
np.savetxt('test.out', x)