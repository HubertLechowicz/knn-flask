import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
body = {
    "Pregnancies": 2,
    "Glucose": 150,
    "SkinThickness": 20,
    "BMI": 20.5,
    "Age": 30
}
response = requests.post(url, data=body)
print(response.json())