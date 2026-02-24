import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Can you also read the server.py file and explain what's there"},
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
