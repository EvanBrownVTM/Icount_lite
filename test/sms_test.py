import requests
import datetime
import json

tsv_url = 'http://192.168.1.140:8085/tsv/flashapi'

def sms_text(tsv_url):
	sms_response = requests.post(url= tsv_url, data='["CreateSMSText", "CV ALERT Testing {}"]'.format(datetime.datetime.now().strftime("%c"))).json()
	print(sms_response)

sms_text(tsv_url)
