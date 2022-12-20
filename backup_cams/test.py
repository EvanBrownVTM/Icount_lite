import json
import requests
import traceback

headers = {'Authorization': 'Bearer TH0LYa2hTkMIZAlwPErmoizgDZSW'}
base_url = 'https://viatouchmedia-test.apigee.net'

transid = '520a4bab-19c8-4e17-beba-843cd3291f24'
#fileobj = open('post_archive/{}.zip'.format(transid), 'rb')
fileobj = json.load(open('post_archive/{}/transaction_summary.json'.format(transid), 'r'))
print(fileobj)
#print(json.dumps(fileobj))
try:
        response = requests.put("{}/loyalty/machines/activity".format(base_url), json = fileobj, headers=headers)
        print(response)
        print(response.json())
except Exception as e:
        #logger.info("      Uploading Failed")
        print(traceback.format_exc())

