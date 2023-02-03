import numpy as np
#e1 = np.load('top_shelf.npy')
# e2 = np.load('lowest_shelf.npy')
e3 = np.load('lower_shelf.npy', allow_pickle=True)

np.savez('cam0_Regus_office.npz', all_shelves = e3)

