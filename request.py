import requests

url = 'http://127.0.0.1:5000/predict_api'
r = requests.get(url,json={'age':25, 'marital_status':1, 'income':45000})

print(r.json())
