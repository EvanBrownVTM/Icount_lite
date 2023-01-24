import numpy as np
import cv2

"""
e = np.load('contours_v2.npz', allow_pickle=True)
img = cv2.imread('test2.jpg')


'''
raw_larger = np.int32(e['larger'] * 640)
raw_larger[:,:,-1] -= 5
#larger = np.int32(e['larger'] * 640)
#print(larger)
raw_smaller = np.int32(e['smaller'] * 640)
raw_smaller[:,:,-1] -= 5

np.savez('contours_v2.npz', larger = raw_larger / 640, smaller = raw_smaller / 640)
'''

raw_larger = np.int32(e['larger'] * 640)
raw_smaller = np.int32(e['smaller'] * 640)

cv2.drawContours(img, raw_larger, -1, (0,0,255), 2)
cv2.drawContours(img, raw_smaller, -1, (0,255,0), 2)

cv2.imwrite('test4.jpg', img)
"""
'''
e = np.load('zones_contours_updated_v3.npz', allow_pickle=True)
print(e.files)


img = cv2.imread('res.jpg')
second_shelf = np.int32(e['second_shelf'] * 640)
#second_shelf[:,:,0] += 25
cv2.drawContours(img, [second_shelf], 0, (0,0,255), 2)

lower_shelves = np.int32(e['lower_shelves'] * 640)
#lower_shelves[:,:,0] += 25
cv2.drawContours(img, [lower_shelves], 0, (0,255,0), 2)

top_shelf = np.int32(e['top_shelf'] * 640)
#top_shelf[:,:,0] += 25
cv2.drawContours(img, [top_shelf], 0, (255,0,0), 2)

cv2.imwrite('test2.jpg', img)

#np.savez('zones_contours_updated_v3.npz', second_shelf = second_shelf / 640, lower_shelves = lower_shelves / 640, top_shelf = top_shelf / 640)

'''



e = np.load('zones2_contours_v6.npz', allow_pickle=True)
print(e.files)


img = cv2.imread('check_img_resized_cam2.jpg')
lower_shelf = np.int32(e['lower_shelf'] * 640)
#lower_shelf[:,:,0] += 30
cv2.drawContours(img, [lower_shelf], 0, (0,0,255), 2)

lowest_shelf = np.int32(e['lowest_shelf'] * 640)
#lowest_shelf[:,:,0] += 30
cv2.drawContours(img, [lowest_shelf], 0, (0,255,0), 2)

#np.savez('zones2_contours_v3.npz', lower_shelf = lower_shelf / 640, lowest_shelf = lowest_shelf / 640)

cv2.imwrite('test4.jpg', img)


'''
e = np.load('contour2_v2.npy')
t = np.int32(e * 640)
print(np.shape(t))
img = cv2.imread('cam2.jpg')
cv2.drawContours(img, [t], 0, (0,0,255), 2)

cv2.imwrite('cam2_2.jpg', img)

'''


