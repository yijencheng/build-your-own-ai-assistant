import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={"message": "Can you you read what's in README for me?"},
)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
