import requests

url = "http://127.0.0.1:5000/predict"
file_path = "test_leaf.jpg"  # must exist in backend folder

with open(file_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print("🔍 Status code:", response.status_code)
print("📩 Raw response:", response.text)  # print raw server reply

try:
    print("✅ JSON:", response.json())
except Exception as e:
    print("⚠️ Could not parse JSON:", e)
