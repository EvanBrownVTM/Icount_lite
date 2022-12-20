import os
import sys
import time
import logging
import pika
import copy
import numpy as np
import cv2
import json
import pycuda.autoinit  # This is needed for initializing CUDA driver
import multiprocessing
import utils_lite.configSrc as cfg

from pypylon import pylon
from collections import deque, Counter, defaultdict
from utils.yolo_with_plugins import TrtYOLO
from utils.display import open_window, set_display, show_fps
from utils.visualization_ic import BBoxVisualization
from utils_lite.tracker import AVT
from utils_lite.side_cam_solver import SideCam
from utils_lite.front_cam_solver import FrontCam

from  utils_lite.utils import descale_contour

fps = 0.0
conf_th = 0.6
maxCamerasToUse = 3
archive_size = 416
#cls_dict = {0:'goli_red', 1:'goli_blue', 2:'clinique', 3:'mtn', 4:'pepsi'}
cls_dict = {0:'Mtn', 1:'Pepsi', 2:'Peanut', 3:'Sour'}
display_mode = False
cam0_delay = 0
cam_id = 'cam0'
pika_flag = False
l_mask = np.load('utils_lite/lowest_shelf_mask.npy')
l_mask = np.int32(l_mask * archive_size)

def init():
	print('\tLoading TensoRT model...')
	# build the class (index/name) dictionary from labelmap file
	trt_yolo = TrtYOLO("yolov4-tiny-416", (416, 416), 4, False, path_folder = 'yolo/')
	print('\tModel loaded and ready for detection')

	#print('\tRunning warmup detection')
	dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
	_, _, _ = trt_yolo.detect(dummy_img, 0.6)

	return trt_yolo

def initializeCamera(serial_number_list):	
	# Get the transport layer factory.
	pfs_list = ['ic_out_front.pfs', 'ic_side_cam2.pfs']
	tlFactory = pylon.TlFactory.GetInstance()

	# Get all attached devices and exit application if no device is found.
	devices = tlFactory.EnumerateDevices()
	if len(devices) == 0:
		raise pylon.RuntimeException("No camera present.")

	# Create an array of instant cameras for the found devices and avoid exceeding a maximum number of devices.
	cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

	# Create and attach all Pylon Devices.
	for i, cam in enumerate(cameras):
		info = pylon.DeviceInfo()
		info.SetSerialNumber(str(serial_number_list[i]))
		cam.Attach(tlFactory.CreateDevice(info))
		cam.Open()
		pylon.FeaturePersistence.Load(pfs_list[i], cam.GetNodeMap(), True)		
		
	return cameras, len(devices)

#RabbitMQ Initialization
def initializeChannel():
	#Initialize queue for door signal
	credentials = pika.PlainCredentials('nano','nano')
	parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials, blocked_connection_timeout=3000)
	connection = pika.BlockingConnection(parameters)
	channel = connection.channel()
	channel.queue_declare(queue='cvRequest',durable = True)

	#Clear queue for pre-existing messages 
	channel.queue_purge(queue='cvRequest')
	
	#logger.info("   Rabbitmq connections initialized ")
	return channel, connection


def trt_detect(frame, trt_yolo, conf_th, vis):
	if frame is not None:
		boxes, confs, clss = trt_yolo.detect(frame, conf_th)
		if display_mode:
			frame = vis.draw_bboxes(frame, boxes, confs, clss)
		return frame, clss, boxes


def update_logic(new_boxes, clss, frame, cam_solver, avt, frame_id,frame_draw):
	cents = []
	cent2bbox = {}
	cent2cls = {}
	id2active_zone = {}
	
	for i in range(len(new_boxes)):
		bbox = new_boxes[i]
		cls = clss[i]
		cents.append([(bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2])
		cent2bbox["{}_{}".format((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)] = bbox
		cent2cls["{}_{}".format((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)] = cls
		
	objects, disappeared = avt.update(cents)

	for (objectID, centroid) in objects.items():		
		cent_symbol = "{}_{}".format(centroid[0], centroid[1])
		if cent_symbol not in cent2bbox:
			continue
		cam_solver.update_tracks(cent2cls[cent_symbol], cent2bbox[cent_symbol], objectID, frame_id, frame)
		id2active_zone[objectID] = cam_solver._tracks[objectID]._active_zone
		
		if display_mode:
			text = "ID {}, {}".format(str(objectID), cam_solver._tracks[objectID]._active_zone)
			cv2.putText(frame_draw, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
		
	return id2active_zone

def solver_infer(cam_solver, idle_flag = False):
	cam_solver.interact_tracks(idle_flag)

def merge_cart(cam0_solver, cam2_solver):
	cart = defaultdict(int)
	for cl in cam0_solver.cart:
		cart[cl] += cam0_solver.cart[cl]
	
	for cl in cam2_solver.cart:
		cart[cl] += cam2_solver.cart[cl]
	
	return cart
		

def displayCart(det_frame, cart):
	#cv2.putText(det_frame, 'Cart:', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cnt = 0
	for prod_ind in sorted(cart):
		if cart[prod_ind] != 0:
			cv2.putText(det_frame, "{}:{}".format(cls_dict[prod_ind], cart[prod_ind]), (0, 50  + 30 * cnt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			cnt += 1
	
def infer_engine(frame0, frame2, frame_cnt0, frame_cnt2):
	global cam0_delay, cam_id
	frame0_copy = frame0.copy()
	frame2_copy = frame2.copy()
	
	det_frame0, clss0, new_boxes0 = trt_detect(frame0, trt_yolo, conf_th, vis)
	det_frame2, clss2, new_boxes2 = trt_detect(frame2, trt_yolo, conf_th, vis)

	id2active_zone0 = update_logic(new_boxes0, clss0, frame0_copy, cam0_solver, avt0, frame_cnt0 - 1, frame0)
	id2active_zone2 = update_logic(new_boxes2, clss2, frame2_copy, cam2_solver, avt2, frame_cnt2 - 1, frame2)

	'''
	if 'top_shelf' in id2active_zone1.values():
		cam0_delay = 50
		cam_id = 'cam1'
		
	elif 'second_shelf' in id2active_zone1.values() or 'lower_shelves' in id2active_zone1.values():
		cam0_delay = 0
		cam_id = 'cam0'
		
	else:
		if 'lowest_shelf' in id2active_zone2.values() or 'lower_shelves' in id2active_zone2.values():
			cam0_delay = 50
			cam_id = 'cam2'
		else:
			if cam0_delay > 0:
				cam0_delay -= 1
			if cam0_delay == 0:
				cam_id = 'cam0'
	#print(cam_id)
	'''
	'''
	if cam_id == 'cam0':
		solver_infer(cam0_solver)
		solver_infer(cam1_solver, True)
		solver_infer(cam2_solver, True)
		
	elif cam_id == 'cam2':
		solver_infer(cam0_solver, True)
		solver_infer(cam1_solver, True)
		solver_infer(cam2_solver)
	'''
	
	solver_infer(cam0_solver)
	solver_infer(cam2_solver)
	
	cart = merge_cart(cam0_solver, cam2_solver)
	#print(cart)
	
	
	return det_frame0, det_frame2, cart

if pika_flag:
	channel, connection = initializeChannel()

avt0 = AVT()
avt2 = AVT()

trt_yolo = init() 
vis = BBoxVisualization(cls_dict)

cam0_solver = FrontCam('cam0', 'utils_lite/machine_zones_v3.npz')
cam2_solver = SideCam('cam2', 'utils_lite/zones2_contours_v5.npz')
	
	
tic = time.time()

cameras, dev_len = initializeCamera([cfg.camera_map["cam0"], cfg.camera_map["cam0_2"]])
#cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
grabbing_status = 0
frame_cnt0 = 0
frame_cnt2 = 0
check_list = [ False for i in range(dev_len)]
if pika_flag:
	door_state = 'Init'
else:
	door_state = 'DoorOpened'
	cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

while True:
	if pika_flag:
		_,_,recv = channel.basic_get('cvRequest')
		if recv != None:
			recv = str(recv,'utf-8')
			recv =json.loads(recv)

			if recv["cmd"] == 'DoorOpened':
				transid = recv["parm1"]
				print("   RECV: {} / cvRequest".format(recv["cmd"]))
				door_state = "DoorOpened"
				if grabbing_status == 0:
					cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
					grabbing_status = 1
				
			elif recv["cmd"] == 'DoorLocked':
				transid = recv["parm1"]
				print("   RECV: {} / cvRequest".format(recv["cmd"]))
				door_state = "DoorLocked"
				if grabbing_status == 1:
					cameras.StopGrabbing()
					grabbing_status = 0
			
	if door_state == "DoorOpened":
		if cameras.IsGrabbing():
			try:
				grabResult = cameras.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
			except:
				print("Camera disconnected")
				cameras.Close()
				cameras, dev_len = initializeCamera([cfg.camera_map["cam0"], cfg.camera_map["cam0_2"]])
				check_list = [ False for i in range(dev_len)]
				cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
				continue
				
			cameraContextValue = grabResult.GetCameraContext()
			if cameraContextValue == 0:
				frame_cnt0 += 1
			else:
				frame_cnt2 += 1
				
			if grabResult.GrabSucceeded():
				if cameraContextValue == 0:
					frame0 = cv2.resize(np.uint8(grabResult.Array), (archive_size, archive_size))
					check_list[0] = True
					
				else:
					frame2 = cv2.resize(np.uint8(grabResult.Array), (archive_size, archive_size))
					frame2 = cv2.rotate(frame2, cv2.ROTATE_90_COUNTERCLOCKWISE)
					#cv2.drawContours(frame2, [l_mask], 0, (0,0,0), -1)
					check_list[1] = True


			if all(check_list):
				check_list = np.logical_not(check_list)
				
				det_frame0, det_frame2, cart = infer_engine(frame0, frame2, frame_cnt0, frame_cnt2)
				
				if display_mode:
					img_hstack = det_frame0
					img_hstack = np.hstack((img_hstack, det_frame2))
					img_hstack = show_fps(img_hstack, fps)
					displayCart(img_hstack, cart)
					
					cv2.imshow('Yo', img_hstack[:,:,::-1])
					
					key = cv2.waitKey(1)
					if key == 27:  # ESC key: quit program
						break
				
				toc = time.time()
				curr_fps = 1.0 / (toc- tic)
				fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
				tic = toc
				if frame_cnt0 % 100 == 0:
					print(fps)
				
			grabResult.Release()
	

