import requests

response = requests.post(
    "http://localhost:4000/predict", 
    json={
        "model_key":"Peugeot",
        "mileage":100000,
        "engine_power":110,
        "fuel":"diesel",
        "paint_color":"black",
        "car_type":"sedan",
        "private_parking_available":False,
        "has_gps":True,
        "has_air_conditioning":True,
        "automatic_car":False,
        "has_getaround_connect":False,
        "has_speed_regulator":True,
        "winter_tires":False
})

print(response.json())