import requests
query = """
[out:json][timeout:15];
(
  nwr["amenity"="hospital"](around:50000,28.7041,77.1025);
  nwr["amenity"="clinic"](around:50000,28.7041,77.1025);
  nwr["amenity"="doctors"](around:50000,28.7041,77.1025);
  nwr["healthcare"="hospital"](around:50000,28.7041,77.1025);
  nwr["healthcare"="clinic"](around:50000,28.7041,77.1025);
  nwr["healthcare"="centre"](around:50000,28.7041,77.1025);
  nwr["emergency"="ambulance_station"](around:50000,28.7041,77.1025);
);
out center;
"""
url = "https://overpass-api.de/api/interpreter"
r = requests.post(url, data={"data": query}, headers={'User-Agent': 'TrafficAccidentPredictionApp/1.0'}, timeout=15)
print(r.status_code)
if r.status_code == 200:
    data = r.json()
    print("Found elements:", len(data.get('elements', [])))
else:
    print(r.text)
