import numpy as np
#e1 = np.load('top_shelf.npy')
e2 = np.load('lowest_shelf.npy')
e3 = np.load('lower_shelf.npy')

np.savez('zones2_contours_v6.npz', lowest_shelf = e2, lower_shelf = e3)

