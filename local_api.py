import requests

# Define the base URL for the API
base_url = "http://127.0.0.1:8000"

# Send a GET request to the root endpoint
response = requests.get(base_url)
print("GET request:")
print("Status Code:", response.status_code)
try:
    # Extract "message" key for a cleaner output
    print("Result:", response.json().get("message", "No message found"))
except requests.JSONDecodeError:
    print("Result: Could not parse JSON. Raw response:", response.text)

# Sample data for the POST request
sample_data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

# Send a POST request to the /predict/ endpoint
response = requests.post(f"{base_url}/predict/", json=sample_data)
print("\nPOST request:")
print("Status Code:", response.status_code)
try:
    # Extract "result" key to display only the prediction result
    print("Result:", response.json().get("result", "No result found"))
except requests.JSONDecodeError:
    print("Result: Could not parse JSON. Raw response:", response.text)
