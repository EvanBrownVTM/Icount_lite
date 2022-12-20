#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvRequest",durable = True)

data = '{\n "cmd": "DoorLocked", \n "parm1":"trans702"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvRequest",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()


'''
credentials = pika.PlainCredentials('nano','nano')
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
