"""
Test to find a working Hugging Face endpoint
"""
import requests
import json

print("Testing Hugging Face endpoints...\n")

# Test 1: Try Inference Endpoints (newer API)
print("1. Testing Hugging Face Inference Endpoints API...")
try:
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Content-Type": "application/json"}
    payload = {
        "inputs": "a simple black cat silhouette on white background",
        "parameters": {
            "width": 512,
            "height": 512,
            "num_inference_steps": 10
        }
    }
    print(f"   URL: {url}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   [SUCCESS] Got image!")
        with open("test_hf_image.png", "wb") as f:
            f.write(response.content)
        print("   Saved to test_hf_image.png")
    else:
        print(f"   Error: {response.text[:300]}")
except Exception as e:
    print(f"   Exception: {e}")

# Test 2: Try router endpoint
print("\n2. Testing Hugging Face Router endpoint...")
try:
    url = "https://router.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Content-Type": "application/json"}
    payload = {"inputs": "a simple black cat silhouette on white background"}
    print(f"   URL: {url}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   [SUCCESS] Got image!")
        with open("test_router_image.png", "wb") as f:
            f.write(response.content)
        print("   Saved to test_router_image.png")
    else:
        print(f"   Error: {response.text[:300]}")
except Exception as e:
    print(f"   Exception: {e}")

# Test 3: Try Inference Endpoints API (different format)
print("\n3. Testing Hugging Face Inference Endpoints (alternative format)...")
try:
    url = "https://inference-endpoints.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Content-Type": "application/json"}
    payload = {"inputs": "a simple black cat silhouette on white background"}
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   [SUCCESS] Got image!")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   Exception: {e}")

print("\nDone!")

