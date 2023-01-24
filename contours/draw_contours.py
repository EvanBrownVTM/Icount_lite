import cv2
import numpy as np 

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None
glist = []
f = open('lower_shelf.npy',  'wb')
# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing
    if drawing == True:
        glist.append([[x / 640, y / 640]])
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=3)        


#img = np.zeros((512,512,3), np.uint8)
img = cv2.imread('check_img_resized_cam2.jpg')
cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)

while(1):
    cv2.imshow('test draw',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
print(np.shape(glist))
np.save(f, np.asarray(glist))
