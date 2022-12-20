import os
import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("/usr/local/cuda-10.2/bin")
sys.path.append("/usr/local/cuda-10.2/lib64")
import time
import logging
import pika
import copy
import numpy as np
import cv2
import json
import pycuda.autoinit  # This is needed for initializing CUDA driver
import utils_lite.configSrc as cfg
import tensorflow as tf

from pypylon import pylon
from collections import deque, Counter, defaultdict
from utils.yolo_with_plugins import TrtYOLO
from utils.display import open_window, set_display, show_fps
from utils.visualization_ic import BBoxVisualization
from utils_lite.tracker import AVT
from utils_lite.front_cam_solver import FrontCam
from  utils_lite.utils import descale_contour

logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(filename='/home/cvnx/Desktop/logs/Icount.log', level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
logging.disable(logging.DEBUG)
logger=logging.getLogger()
logger.info("")
sys.stderr.write=logger.error

init_process = True
timestamp_format = "%Y%m%d-%H_%M_%S"
archive_flag = True
fps = 0.0
conf_th = 0.7
maxCamerasToUse = 1
archive_size = 416
#cls_dict = {0:'goli_red', 1:'goli_blue', 2:'clinique', 3:'mtn', 4:'pepsi'}
cls_dict = {0:'Mtn', 1:'Pepsi', 2:'Peanut', 3:'Sour'}
display_mode = False
cam0_delay = 0
cam_id = 'cam0'
pika_flag = True



def init():
	logger.info('Loading TensoRT model...')
	# build the class (index/name) dictionary from labelmap file
	trt_yolo = TrtYOLO("yolov4-tiny-416", (416, 416), 4, False, path_folder = 'yolo/')

	#print('\tRunning warmup detection')
	dummy_img = np.zeros((416, 416, 3), dtype=np.uint8)
	_, _, _ = trt_yolo.detect(dummy_img, 0.6)
	logger.info('Model loaded and ready for detection')
	
	return trt_yolo

def initializeCamera(serial_number_list):	
	# Get the transport layer factory.
	pfs_list = ['ic_out_front.pfs']
	tlFactory = pylon.TlFactory.GetInstance()

	# Get all attached devices and exit application if no device is found.
	devices = tlFactory.EnumerateDevices()
	if len(devices) == 0:
		logger.info("No Camera Detected")
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
	logger.info("Camera Initialized")
	return cameras, len(devices)

#RabbitMQ Initialization
def initializeChannel():
	#Initialize queue for door signal
	credentials = pika.PlainCredentials('nano','nano')
	parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials, blocked_connection_timeout=3000)
	connection = pika.BlockingConnection(parameters)
	channel = connection.channel()
	channel.queue_declare(queue='cvRequest',durable = True)
	
	channel2 = connection.channel()
	channel2.queue_declare(queue='cvPost',durable = True)

	#Clear queue for pre-existing messages 
	channel.queue_purge(queue='cvRequest')
	channel2.queue_purge(queue='cvPost')
	
	logger.info("Rabbitmq connections initialized ")
	return channel, channel2, connection


def trt_detect(frame, trt_yolo, conf_th, vis):
	if frame is not None:
		boxes, confs, clss = trt_yolo.detect(frame, conf_th)
		if display_mode:
			frame = vis.draw_bboxes(frame, boxes, confs, clss)
		return frame, clss, boxes, confs


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

def solver_infer(cam_solver, logger, cv_activities, idle_flag = False):
	cam_solver.interact_tracks(logger, cv_activities, idle_flag)

def merge_cart(cam0_solver):
	cart = defaultdict(int)
	for cl in cam0_solver.cart:
		cart[cl] += cam0_solver.cart[cl]
	
	return cart
		

def displayCart(det_frame, cart):
	#cv2.putText(det_frame, 'Cart:', (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cnt = 0
	for prod_ind in sorted(cart):
		if cart[prod_ind] != 0:
			cv2.putText(det_frame, "{}:{}".format(cls_dict[prod_ind], cart[prod_ind]), (0, 50  + 30 * cnt), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
			cnt += 1
	
def infer_engine(frame0, frame_cnt0, timestr0, cv_activities):
	global cam0_delay, cam_id
	frame0_copy = frame0.copy()
	
	det_frame0, clss0, new_boxes0, confs0 = trt_detect(frame0, trt_yolo, conf_th, vis)
	file2info = {}
	file2info['bboxes'] = np.asarray(new_boxes0, dtype=np.int32).tolist()
	file2info['classes'] = np.asarray(clss0, dtype = np.int32).tolist()
	file2info['scores'] = np.asarray(confs0).tolist()
	if not os.path.exists("{}archive/{}/cam0/prod".format(cfg.base_path, transid)):
		os.makedirs("{}archive/{}/cam0/prod".format(cfg.base_path, transid))
	
	f_name = "%s_%05d"%(timestr0, int(frame_cnt0))
	
	json.dump(file2info, open('{}archive/{}/cam0/prod/{}.json'.format(cfg.base_path, transid, f_name), 'w'))

	
	id2active_zone0 = update_logic(new_boxes0, clss0, frame0_copy, cam0_solver, avt0, frame_cnt0 - 1, frame0)
	solver_infer(cam0_solver, logger, cv_activities)
	
	cart = merge_cart(cam0_solver)
	
	return det_frame0, cart

#convert raw image to bytes
def _bytes_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() 
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#compress image bytes 
def img2jpeg(image):
	is_success, im_buf_arr = cv2.imencode(".jpg", image)
	byte_im = im_buf_arr.tobytes()
	return byte_im

if pika_flag:
	channel, channel2, connection = initializeChannel()


avt0 = AVT()

trt_yolo = init() 
vis = BBoxVisualization(cls_dict)

cam0_solver = FrontCam('cam0', 'utils_lite/machine_zones_v3.npz')

tic = time.time()

cameras, dev_len = initializeCamera([cfg.camera_map["cam0"]])
grabbing_status = 0
frame_cnt0 = 0
act_flag = 0
transid = 'trans_init'
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
			#print(recv)
			recv = str(recv,'utf-8')
			recv =json.loads(recv)
			if recv["cmd"] == 'DoorOpened':
				transid = recv["parm1"].split(":")[0]
				door_info = recv["parm1"].split(":")[1]
				logger.info("   RECV: {} / cvRequest".format(recv["cmd"]))
				logger.info("      TRANSID: {}".format(transid))
				door_state = "DoorOpened"
				frame_cnt0 = 0
				cv_activities = []
				if grabbing_status == 0 and door_info == 'True':
					cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
					grabbing_status = 1
				
			elif recv["cmd"] == 'DoorLocked':
				transid = recv["parm1"]
				logger.info("   RECV: {} / cvRequest".format(recv["cmd"]))
				logger.info("      TRANSID: {}".format(transid))
				door_state = "DoorLocked"
				if grabbing_status == 1:
					cameras.StopGrabbing()
					grabbing_status = 0
			elif recv["cmd"] == "ActivityID":
				ls_activities = recv["parm1"]
				act_flag = 1
	if door_state == "DoorOpened":
		clear_flag = 1
		if cameras.IsGrabbing():
			try:
				grabResult = cameras.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
			except:
				logger.info("Camera Disconnected")
				cameras.Close()
				cameras, dev_len = initializeCamera([cfg.camera_map["cam0"]])
				check_list = [ False for i in range(dev_len)]
				cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
				continue
				
			cameraContextValue = grabResult.GetCameraContext()
			
			if grabResult.GrabSucceeded():
				if cameraContextValue == 0:
					frame_cnt0 += 1
					frame0 = cv2.resize(np.uint8(grabResult.Array), (archive_size, archive_size))
					check_list[0] = True
					timestr = time.strftime(timestamp_format)
					if archive_flag:
						data = {
							  'bytes': _bytes_feature(value = img2jpeg(frame0)), 
							  'timestamp': _bytes_feature(value = timestr.encode('utf-8'))
							}
							
						features = tf.train.Features(feature=data)
						example = tf.train.Example(features=features)
						serialized = example.SerializeToString()
						
						
						if init_process == True:
							if not os.path.exists("{}archive/{}".format(cfg.base_path, transid)):
								os.mkdir("{}archive/{}".format(cfg.base_path, transid))
							writer0 = tf.python_io.TFRecordWriter("{}archive/{}/img_0.tfrecords".format(cfg.base_path, transid))
							init_process = False
							
						writer0.write(serialized)
					
						
			if all(check_list):
				check_list = np.logical_not(check_list)
				
				det_frame0, cart = infer_engine(frame0, frame_cnt0, timestr, cv_activities)
				
				if display_mode:
					img_hstack = det_frame0
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
				#if frame_cnt0 % 100 == 0:
					#print(fps)
				time.sleep(0.005)
			grabResult.Release()
	

	elif door_state == "DoorLocked" and clear_flag == 1:
		if archive_flag and door_info == 'True':
			writer0.close()
			init_process = True
		clear_flag = 0

	elif door_state == "DoorLocked" and act_flag == 1:
		data = {"cmd": "Done", "transid": transid, "timestamp": time.strftime("%Y%m%d-%H_%M_%S"), "cv_activities": cv_activities, "ls_activities": ls_activities}
		mess = json.dumps(data)
		
		channel2.basic_publish(exchange='',
						routing_key="cvPost",
						body=mess)
		
		logger.info("Sent cvPost signal\n")
		
		door_state = 'initialize'
		act_flag = 0
		ls_activities = ""
		
		
