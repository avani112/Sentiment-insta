import requests
url = 'http://localhost:5000/results'
r = requests.post(url, json={'short_code':"", 'max_comments':""})

print(r.json())