#!/usr/bin/env python
import pika
import json
import sys
sys.path.insert(0, '../')
import configSrc as cfg

credentials = pika.PlainCredentials(cfg.pika_name,cfg.pika_name)
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvIcount",durable = True)

data = '{\n "cmd": "DoorLocked", \n "parm1":"trans702"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvIcount",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()


'''
credentials = pika.PlainCredentials(cfg.pika_name,cfg.pika_name)
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvArchive1",durable = True)

data = '{\n "cmd": "DoorLocked", \n "parm1":"trans1"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvArchive1",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
'''
