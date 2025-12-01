"""
Test script to verify AI image generation is working
Run this before starting the Flask server to diagnose issues
"""

import os
import sys

print("=" * 60)
print("AI Image Generation Test")
print("=" * 60)

# Check environment variables
openai_key = os.environ.get('OPENAI_API_KEY', '')
hf_key = os.environ.get('HUGGINGFACE_API_KEY', '')

print(f"\n1. Environment Variables:")
print(f"   OPENAI_API_KEY: {'Set' if openai_key else 'NOT SET'}")
if openai_key:
    print(f"      Preview: {openai_key[:20]}...")
print(f"   HUGGINGFACE_API_KEY: {'Set' if hf_key else 'NOT SET'}")

# Test OpenAI
print(f"\n2. Testing OpenAI DALL-E...")
if openai_key:
    try:
        import openai
        client = openai.OpenAI(api_key=openai_key)
        print("   [OK] OpenAI library imported")
        
        # Try a simple API call
        print("   [INFO] Attempting to generate test image...")
        response = client.images.generate(
            model="dall-e-3",
            prompt="a simple black circle on white background",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        print("   [SUCCESS] OpenAI DALL-E is working!")
        print(f"   Image URL: {response.data[0].url}")
    except ImportError:
        print("   [ERROR] OpenAI library not installed")
        print("   Run: pip install openai")
    except Exception as e:
        print(f"   [ERROR] OpenAI failed: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
else:
    print("   [SKIP] OpenAI API key not set")

# Test Hugging Face
print(f"\n3. Testing Hugging Face API...")
try:
    import requests
    
    # Test with a simple model
    test_url = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
    headers = {"Content-Type": "application/json"}
    if hf_key:
        headers["Authorization"] = f"Bearer {hf_key}"
    
    payload = {
        "inputs": "a simple black circle on white background",
        "parameters": {
            "width": 256,
            "height": 256,
            "num_inference_steps": 10
        }
    }
    
    print("   [INFO] Attempting to connect to Hugging Face...")
    response = requests.post(test_url, headers=headers, json=payload, timeout=30)
    
    if response.status_code == 200:
        print("   [SUCCESS] Hugging Face API is working!")
    elif response.status_code == 503:
        print("   [INFO] Model is loading (this is normal for first request)")
        print("   [INFO] Wait 10-30 seconds and try again")
    else:
        print(f"   [ERROR] Hugging Face returned status {response.status_code}")
        error_text = response.text[:200] if hasattr(response, 'text') else "Unknown"
        print(f"   Error: {error_text}")
        
except ImportError:
    print("   [ERROR] requests library not installed")
    print("   Run: pip install requests")
except Exception as e:
    print(f"   [ERROR] Hugging Face test failed: {str(e)}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
print("\nIf OpenAI or Hugging Face tests failed:")
print("1. Make sure API keys are set in your environment")
print("2. Check your internet connection")
print("3. Verify API keys are valid")
print("4. For Hugging Face, first request may take time (model loading)")

