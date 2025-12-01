"""
Test script to find a working AI image generation API
"""
import requests
import json

print("Testing different AI APIs...\n")

# Test 1: Hugging Face Inference API (direct)
print("1. Testing Hugging Face Inference API...")
try:
    url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Content-Type": "application/json"}
    payload = {
        "inputs": "a simple black cat silhouette on white background",
        "parameters": {"width": 512, "height": 512, "num_inference_steps": 10}
    }
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print("   [SUCCESS] Hugging Face works!")
        with open("test_image.png", "wb") as f:
            f.write(response.content)
        print("   Saved test image to test_image.png")
    else:
        print(f"   Error: {response.text[:200]}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Try a simpler Hugging Face endpoint
print("\n2. Testing alternative Hugging Face endpoint...")
try:
    url = "https://huggingface.co/api/models/runwayml/stable-diffusion-v1-5"
    response = requests.get(url, timeout=10)
    print(f"   Status: {response.status_code}")
except Exception as e:
    print(f"   Error: {e}")

print("\nDone!")

