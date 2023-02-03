'''
draw all contours for each camera in one shot
    -top to bottom
    -press Esc to move on to next contour
fuses and saves as .npz
'''
import cv2
import numpy as np 

def draw(output_contour_path, input_image):
    if isinstance(input_image, str):
        img = cv2.imread(input_image)
    else:
        img = input_image
    global drawing
    drawing = False # true if mouse is pressed
    pt1_x , pt1_y = None , None
    glist = []
    f = open(output_contour_path,  'wb')
    # mouse callback function
    def line_drawing(event,x,y,flags,param):
        global pt1_x,pt1_y, drawing
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

    cv2.namedWindow('contours')
    cv2.setMouseCallback('contours',line_drawing)

    while(1):
        cv2.imshow('contours',img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: #ESC (end drawing)
            break
    cv2.destroyAllWindows()
    np.save(f, np.asarray(glist))
    return img
    

if __name__ == '__main__':
    output_contour_path = 'cam2_Regus_office.npz'
    input_image_path = 'cam2_Regus_office.jpg'
    # shelf_names = ['all_shelves'] #cam0
    # shelf_names = ['top_shelf', 'second_shelf', 'lower_shelves'] #cam1
    shelf_names = ['lower_shelf', 'lowest_shelf'] #cam2

    #draw one contour per shelf
    img = draw(shelf_names[0]+'.npy', input_image_path)
    for shelf in shelf_names[1:]:
        img = draw(shelf+'.npy', img)
    shelf_dict = {shelf:np.load(shelf+'.npy') for shelf in shelf_names}

    #fuse em
    np.savez(output_contour_path, **shelf_dict)

