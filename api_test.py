import requests
import json
import time

url = "http://localhost:8000/predict"
data = {
    "airline": "Vistara",
    "source_city": "Delhi",
    "destination_city": "Mumbai",
    "departure_time": "Morning",
    "arrival_time": "Afternoon",
    "duration": 2.5,
    "days_left": 10,
    "stops": "zero",
    "flight_class": "Economy"
}

print(f"Testing {url} ...")
t0 = time.time()
try:
    res = requests.post(url, json=data)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {json.dumps(res.json(), indent=2)}")
except Exception as e:
    print(f"Failed: {e}")
print(f"Total time taken: {time.time()-t0:.4f}s")
