import os
import sys
sys.path.append("/usr/lib/python3.6/site-packages/")
sys.path.append("/usr/local/cuda-10.2/bin")
sys.path.append("/usr/local/cuda-10.2/lib64")
sys.path.append("/home/cv002/protobuf/bin")
import io
import cv2
import pika
import copy
import json
import time
import logging
import datetime
import requests
import traceback
import numpy as np
#import pycuda.autoinit
import tensorflow as tf
import utils_lite.configSrc as cfg

from PIL import Image
from pypylon import pylon
from collections import Counter
from utils.display import show_fps
from utils.yolo_classes import get_cls_dict
from utils.yolo_with_plugins import TrtYOLO
from utils.visualization_ic import BBoxVisualization




#Customizable Variables
vis_flag = False  #visulization flag
debug_flag = True #debug flag for icount logic
enable_cam = True #camera flag
archive_flag = True #archive flag
frame_cycle = 1 #cycle of archiving frames
init_process = True #archive process initialization flag
trigger_switch_flag = False #weight trigger(switch) flag
time_sleep = 14 #interval for weight trigger voice alert
archive_size = 416 #resolution of image saved in archive
compress_flag = True

#Variable initialization
transid = "initialize"
recv = "initialize"
recv_adc = "initialize"
door_state = "initialize"



#Logger initialization
logging.getLogger("pika").setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(filename='{}logs/Icount.log'.format(cfg.log_path), level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
logging.disable(logging.DEBUG)
logger=logging.getLogger()
logger.info("")
sys.stderr.write=logger.error


#Constant Variables
vicki_app = "http://192.168.1.140:8085/tsv/flashapi"
vicki_headers = {'Content-type': 'application/json'}
timestamp_format = "%Y%m%d-%H_%M_%S"

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)




#Basler Camera Initialization
def initializeCamera(logger, serial_number):
	info = pylon.DeviceInfo()
	info.SetSerialNumber(str(serial_number))
	img = pylon.PylonImage()
	logger.info("   Basler service started ")
	#Camera Initialization
	camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(info))
	camera.Open()
	
	pylon.FeaturePersistence.Load("{}ic_out_front.pfs".format(cfg.base_path), camera.GetNodeMap(), True)
	
	camera.MaxNumBuffer = 15
	camera.DeviceLinkThroughputLimitMode.SetValue('Off')
	camera.GetDeviceInfo().SetPropertyValue("MaxTransferSize", '4194304')
	
	logger.info("   Camera initialized")
	return camera, img

#RabbitMQ Initialization
def initializeChannel(logger):
	#Initialize queue for door signal
	credentials = pika.PlainCredentials('nano','nano')
	parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials, blocked_connection_timeout=3000)
	connection = pika.BlockingConnection(parameters)
	channel = connection.channel()
	channel.queue_declare(queue='cvArchive',durable = True)
	
	connection_adc = pika.BlockingConnection(parameters)
	channel_adc = connection_adc.channel()
	channel_adc.queue_declare(queue='cvADC',durable =True)

	connection_activity = pika.BlockingConnection(parameters)
	channel_activity = connection_activity.channel()
	channel_activity.queue_declare(queue='cvActivity',durable =True)

	#Clear queue for pre-existing messages 
	channel.queue_purge(queue='cvArchive')
	channel_adc.queue_purge("cvADC")
	channel_activity.queue_purge('cvActivity')
	logger.info("   Rabbitmq connections initialized ")
	return channel, channel_adc, channel_activity, connection, connection_adc, connection_activity

	
	
#parser for tfrecords
def parse(serialized):
    features = \
    {
        'bytes': tf.FixedLenFeature([], tf.string),
        'timestamp': tf.FixedLenFeature([], tf.string),
        #'frame_cnt': tf.FixedLenFeature([], tf.string)
    }

    parsed_example = tf.parse_single_example(serialized=serialized,features=features)
    image = parsed_example['bytes']
    timestamp = parsed_example['timestamp']
    #frame_cnt = parsed_example['frame_cnt']
    if compress_flag:
        image = tf.io.decode_image(image)
    else:
        image = tf.decode_raw(image,tf.uint8)
    return {'image':image, 'timestamp':timestamp} #, 'frame_cnt': frame_cnt}



#parse tfrecords to jpg's
def readTfRecords(transid):
	logger.info("      Reading TFRecords")
	dataset = tf.data.TFRecordDataset(["{}archive/three_cams/{}/img_0.tfrecords".format(cfg.base_path, transid)])
	logger.info("      Parsing TFRecords")
	dataset = dataset.map(parse)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	frame_cnt = 0
	logger.info("      Processing Images")
	while True:
		frame_cnt += 1
		try:
			img, timestr = sess.run([next_element['image'], next_element['timestamp']]) #, next_element['frame_cnt']])
			current_frame = img.reshape((archive_size, archive_size, 3))
			#print((archive_size, int(archive_size / 960 * 1280), 3))
			#current_frame = img.reshape((960, 960, 3))
			#current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
			if not os.path.exists("{}archive/three_cams/{}/cam0".format(cfg.base_path, transid)):
				os.mkdir("{}archive/three_cams/{}/cam0".format(cfg.base_path, transid))
			cv2.imwrite('%sarchive/three_cams/%s/cam0/%s_%05d.jpg'%(cfg.base_path, transid, timestr.decode('utf-8'), int(frame_cnt)), current_frame)
		
		except Exception as e:
			print("frame_cnt: ", frame_cnt)
			break

	logger.info("      Reading TFRecords")
	dataset = tf.data.TFRecordDataset(["{}archive/three_cams/{}/img_2.tfrecords".format(cfg.base_path, transid)])
	logger.info("      Parsing TFRecords")
	dataset = dataset.map(parse)
	iterator = dataset.make_one_shot_iterator()
	next_element = iterator.get_next()
	frame_cnt = 0
	logger.info("      Processing Images")

	while True:
		frame_cnt += 1
		try:
			img, timestr = sess.run([next_element['image'], next_element['timestamp']]) #, next_element['frame_cnt']])
			current_frame = img.reshape((archive_size, archive_size, 3))
			'''
			current_frame = img.reshape((int(archive_size / 1280 * 960), archive_size, 3))
			current_frame = cv2.rotate(current_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
			plot = np.zeros((640, 640, 3))
			plot[:,80:80 + 480] = current_frame
			current_frame = plot
			'''
			#print((archive_size, int(archive_size / 960 * 1280), 3))
			#current_frame = img.reshape((960, 960, 3))
			#current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
			if not os.path.exists("{}archive/three_cams/{}/cam0_2".format(cfg.base_path, transid)):
				os.mkdir("{}archive/three_cams/{}/cam0_2".format(cfg.base_path, transid))
			cv2.imwrite('%sarchive/three_cams/%s/cam0_2/%s_%05d.jpg'%(cfg.base_path, transid, timestr.decode('utf-8'), int(frame_cnt)), current_frame)
		
		except Exception as e:
			print("frame_cnt: ", frame_cnt)
			break
			
		
				
				
#convert raw image to bytes
def _bytes_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy() 
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#compress image bytes 
def img2jpeg(image):
	img = Image.fromarray(image)
	img_bytes = io.BytesIO()
	img.save(img_bytes, 'JPEG')
	return img_bytes.getvalue()
        		
			
#Icount Logic - Basic Product Prediction After Transaction
def trigger_logic(transid):
	#print("Post start!")
	t1 = time.strftime(timestamp_format)
	if not os.path.exists("{}archive/{}/cam0".format(cfg.base_path, transid)):
		os.mkdir("{}archive/{}/cam0".format(cfg.base_path, transid))
	readTfRecords(transid, total_cnt)
	logger.info("   Capture Postprocessing Done")
	t2 = time.strftime(timestamp_format)
	info_post = {'transid':transid, 'start':t1, 'end':t2, 'total_frame':total_cnt}
	
	try:
		credentials = pika.PlainCredentials('nano','nano')
		parameters = pika.ConnectionParameters('localhost', 5672, '/', credentials, blocked_connection_timeout=3000)
		connection_post = pika.BlockingConnection(parameters)
		channel_post = connection_post.channel()
		channel_post.queue_declare(queue='cvPost',durable =True)

		channel_post.basic_publish(exchange='', routing_key="cvPost", body=json.dumps(info_post))
		logger.info("   Sent logic trigger signal")
		connection_post.close()
		
	except (pika.exceptions.StreamLostError,
			pika.exceptions.ConnectionClosed,
			pika.exceptions.ChannelClosed,
			pika.exceptions.AMQPConnectionError,
			pika.exceptions.ConnectionWrongStateError,
			ConnectionResetError) as e:

		logger.info("   Failed to establish connection")
		try:
			connection_post.close()
		except:
			pass
		logger.info("	Failed to send trigger signal")
	
	
#Main Function
def main(count):
	logger.info("ICOUNT MODULE (CAPTURE) Activated ")
	global door_state, init_process
	if enable_cam:
		camera, img = initializeCamera(logger, cfg.camera_map["cam0"])
	channel, channel_adc, channel_activity, connection, connection_adc, connection_activity = initializeChannel(logger)
	
	#basic variables - archiving/trans summary
	frame_cnt = 0
	grabbing_status = 0
	fps = 0.0
	time_queue = []
	det_frame = None
	trans_archive = {}
	tic = time.time()
	
	
	#variables - weight trigger alert
	row_number = 5
	row_ind = -1
	pre_time = -1
	clear_flag = 1
	activity_tray_info = ''
	activity_idle_interval = 50
	idle_cnt = 0
	display_message = False
	active_tray = [{} for i in range(row_number)]
	white_light = [set([]) for i in range(row_number)]
	red_light = [set([]) for i in range(row_number)]
	display_stack = []
	activity_tray = []
	activity_tray_event = []
	tray_list = set([])

	
	while True:
		try:
			_,_,recv = channel.basic_get('cvArchive')
			_,_,recv_adc = channel_adc.basic_get('cvADC')
			activity_tray_info = '-1'
			idle_cnt += 1
			if recv != None:
				logger.info("")
				recv = str(recv,'utf-8')
				recv =json.loads(recv)

				if recv["cmd"] == 'DoorOpened':
					channel_adc.queue_purge("cvADC")
					transid = recv["parm1"]
					logger.info("   RECV: {} / cvRequest".format(recv["cmd"]))
					logger.info("      {}".format(transid))
					logger.info('      Received door open signal')
					logger.info("")
					door_state = "DoorOpened"
					trans_archive = {transid:[]}
					if grabbing_status == 0 and enable_cam:
						camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
						grabbing_status = 1
					
					#reinitialize the variables
					start_time = -1
					white_light = [set([]) for i in range(row_number)]
					red_light = [set([]) for i in range(row_number)]
					active_tray = [{} for i in range(row_number)]
					activity_tray = []
					frame_cnt = 0
					
				elif recv["cmd"] == 'DoorLocked':
					transid = recv["parm1"]
					logger.info("   RECV: {} / cvRequest".format(recv["cmd"]))
					logger.info("      {}".format(transid))
					logger.info('      Received door lock signal')
					logger.info("")
					door_state = "DoorLocked"
					with open("{}archive/{}/transaction_summary.json".format(cfg.base_path, transid),'w') as TransJson:
						json.dump(trans_archive, TransJson, indent = 4)
					if grabbing_status == 1 and enable_cam:
						camera.StopGrabbing()
						grabbing_status = 0
					logger.info("   Capture Module Start Postprocessing...")
					#trigger_logic(transid, frame_cnt)
					print("=======================================================")
			if recv_adc != None and recv_adc != "initialize":
				if type(recv_adc) is bytes:
					recv_adc = str(recv_adc,'utf-8')
					recv_adc =json.loads(recv_adc)
					
					if trigger_switch_flag:
						coil_list = recv_adc['coil_ij'].split(',')
						row_ind = int(recv_adc['row']) - 1
						#Light for weight trigger (updated)
						if row_ind != -2:
							white_light[row_ind].clear()
							red_light[row_ind].clear()
							
							# To activate red light
							for coil in coil_list:
								if coil == '':
									continue
								if coil not in active_tray[row_ind]:
									active_tray[row_ind][coil] = False
									red_light[row_ind].add(coil)
								elif not active_tray[row_ind][coil]:
									red_light[row_ind].add(coil)
								
							# To change back to white light
							for coil_ in active_tray[row_ind]:
								if active_tray[row_ind][coil_] and coil_ not in coil_list:
									white_light[row_ind].add(coil_)
									
							
				if 'action' in recv_adc and int(recv_adc['action']) != 0:
					timestr = recv_adc['timestamp']
					timeobj = datetime.datetime.strptime(timestr, timestamp_format)
					location_list = recv_adc['coil_ic'].split(",")
					for loc in location_list:
						activity_piece = {}
						if int(recv_adc['action']) == -1:
							activity_piece["event"] = "PICK"
						else:
							activity_piece["event"] = "RETURN"
						
						activity_piece['location'] = loc
						if activity_piece['location'] not in activity_tray:
							activity_tray.append(activity_piece['location'])
							activity_tray_event = activity_piece['location']
							activity_tray_event = json.dumps(activity_tray_event)
							activity_tray_info = activity_tray_event
							idle_cnt = 0
							activity_tray_event = []
						
						
						activity_piece["timestamp"] = time.strftime(timestamp_format) #timeobj.strftime(timestamp_format) 
						activity_piece['CVProduct'] = "NULL"
						activity_piece['frameIndex'] = frame_cnt
						trans_archive[transid].append(activity_piece)
						time_queue.append([timestr, activity_piece["event"]])
						logger.info("")
						logger.info("\t   EVENT: {}".format(activity_piece["event"]))
						logger.info("\t   LOCATION: {}".format(activity_piece["location"]))
						logger.info("")
		

			if door_state == "DoorOpened":
				clear_flag = 1
				if enable_cam and camera.IsGrabbing():
					grabResult = camera.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
					#print(grabResult.GrabSucceeded())
					if grabResult.GrabSucceeded():
						#frame = np.uint8(grabResult.Array)
						frame = cv2.resize(np.uint8(grabResult.Array), (archive_size, archive_size))
						if compress_flag:
							data = {
							  'bytes': _bytes_feature(value = img2jpeg(frame)), 
							  'timestamp': _bytes_feature(value = time.strftime(timestamp_format).encode('utf-8'))
							}
						else:
							data = {
							  'bytes': _bytes_feature(value = frame.tobytes()),
							  'timestamp': _bytes_feature(value = time.strftime(timestamp_format).encode('utf-8'))
							}
						features = tf.train.Features(feature=data)
						example = tf.train.Example(features=features)
						serialized = example.SerializeToString()
							
						frame_cnt += 1
						det_frame = frame.copy()
						
						#Archiving
						if archive_flag:
							timestr = time.strftime(timestamp_format)
							
							if init_process == True:
								if not os.path.exists("{}archive/{}".format(cfg.base_path, transid)):
									os.mkdir("{}archive/{}".format(cfg.base_path, transid))
								writer = tf.python_io.TFRecordWriter("{}archive/{}/img.tfrecords".format(cfg.base_path, transid))
								init_process = False
								
							writer.write(serialized)
							
						curr_time = time.time()
							
							
							
						toc = time.time()
						curr_fps = 1.0 / (toc - tic)
						# calculate an exponentially decaying average of fps number
						fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
						tic = toc
					grabResult.Release()
						
				
			elif door_state == "DoorLocked" and clear_flag == 1:
				if os.path.exists('{}archive/{}/'.format(cfg.base_path, transid)):
					with open('{}archive/{}/time_sequence.json'.format(cfg.base_path, transid),'w') as tsq:
						json.dump(time_queue, tsq)
				time_queue = []
				door_state = 'initialize'
				det_frame = None
				if archive_flag:
				    writer.close()
				    init_process = True
				cv2.destroyAllWindows()
				tray_list.clear()
				active_tray = [{} for i in range(row_number)]
				white_light = [set([]) for i in range(row_number)]
				red_light = [set([]) for i in range(row_number)]
				display_stack = []
				clear_flag = 0
				
				
			if idle_cnt % activity_idle_interval == 0:
				channel_activity.basic_publish(exchange='', routing_key="cvActivity", body=activity_tray_info)
			  
			if vis_flag == True and (det_frame is not None):
				det_frame = displayCart(det_frame)
				det_frame = show_fps(det_frame, fps)
				cv2.imshow('Yo', cv2.resize(det_frame, (640, 640)))
				key = cv2.waitKey(1)

				if key == 27:  # ESC key: quit program
					break
				elif key == ord('c'):
					clearCart()
					
			#weight trigger alert
			if trigger_switch_flag:
				if row_ind >= 0:
					push_flag = 0
					if len(white_light[row_ind]) > 0:
						for w_coil in white_light[row_ind]:
							if active_tray[row_ind][w_coil]:
								requests.post(url = vicki_app, data = '["setTrayRGB","0", "0", "0", "{}", "{}", "255", "255", "255"]'.format(int(w_coil) // 10, int(w_coil) % 10)).json()
								requests.post(url = vicki_app, data = '["SetCabinetRGB", "255", "255", "255"]').json()
								requests.post(url = vicki_app, data = '["SetTopCabinetRGB", "255", "255", "255"]').json()
								logger.info("\t   Turn on white light @ {}".format(w_coil))
								active_tray[row_ind][w_coil] = False
								if int(w_coil) in tray_list:
									tray_list.remove(int(w_coil))
									
					if len(red_light[row_ind]) > 0:
						for r_coil in red_light[row_ind]:
							if not active_tray[row_ind][r_coil]:
								requests.post(url = vicki_app, data = '["setTrayRGB","0", "0", "0", "{}", "{}", "255", "0", "0"]'.format(int(r_coil) // 10, int(r_coil) % 10)).json()
								requests.post(url = vicki_app, data = '["SetCabinetRGB", "255", "0", "0"]').json()
								requests.post(url = vicki_app, data = '["SetTopCabinetRGB", "255", "0", "0"]').json()
								display_message = True
								push_flag = 1
								tray_list.add(int(r_coil))
								logger.info("\t   Turn on red light @ {}".format(r_coil))
								active_tray[row_ind][r_coil] = True
				
					if len(tray_list) > 0 and push_flag == 1:
						display_stack.append(copy.deepcopy(list(tray_list)))
					
					if display_message:
						trigger_curr_time = time.time()
						if trigger_curr_time - pre_time > time_sleep:
							requests.post(url = vicki_app, data = '["EnableMainLCDChange", "UI_NOTIFY:Product might be misplaced. Check Trays {}:5:#9400D3:#ff0000:playProductMisplaced"]'.format(str(display_stack[0])[1:-1]), headers = vicki_headers)
						else:
							requests.post(url = vicki_app, data = '["EnableMainLCDChange", "UI_NOTIFY:Product might be misplaced. Check Trays {}:5:#9400D3:#ff0000"]'.format(str(display_stack[0])[1:-1]), headers = vicki_headers)
						display_stack.pop(0)
						pre_time = trigger_curr_time
					if len(display_stack) == 0:
						display_message = False
				
		except KeyboardInterrupt:
			logger.info("ICOUNT CAPTURE deactivated")
			connection.close()
			connection_adc.close()
			connection_activity.close()
			sys.exit()

		except Exception as e:
			logger.info(e)
			#logger.info(traceback.format_exc())
			print(traceback.format_exc())
			count += 1
			if count > 2:
				logger.info("Too many restarts - Arch Camera. Report Communication")
				sys.exit()
			connection.close()
			connection_adc.close()
			connection_activity.close()
			if enable_cam:
				camera.StopGrabbing()
				camera.close()
				#updateProcess(method = "pop")
				#updateProcess_match(method = "pop")
				init_process = True
			main(count)
			

	
	cv2.destroyAllWindows()
				
if __name__ == "__main__":
    for transid in os.listdir("archive/three_cams"):
        print("==========================")
        print(transid)
        if not os.path.exists("{}archive/three_cams/{}/cam0".format(cfg.base_path, transid)):
	        os.mkdir("{}archive/three_cams/{}/cam0".format(cfg.base_path, transid))
        readTfRecords(transid)
	    
    #main(0)
	#logger.info("ICOUNT MODULE: Deactivated.")
