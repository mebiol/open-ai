import requests

url1 ='http://192.168.1.100:5001/data'


def send(url):
    while True:    
        im = input('Enter message: ')
        data = {'msg': im}  # Corrected to create a dictionary with a string value
        headers = {'Content-Type': 'application/json'}
        res = requests.post(url, json=data, headers=headers)
        print('---------------')
        print("Status Code:", res.status_code)
        print("Response Text:", res.text)
send(url1)



 