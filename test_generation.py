"""
Quick test to see what's happening with AI generation
"""
import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import generate_2d_image_ai, OPENAI_API_KEY, safe_print

print("=" * 60)
print("Testing AI Image Generation")
print("=" * 60)

print(f"\n1. API Key Status:")
print(f"   OPENAI_API_KEY length: {len(OPENAI_API_KEY)}")
print(f"   OPENAI_API_KEY preview: {OPENAI_API_KEY[:20] if OPENAI_API_KEY else 'NOT SET'}...")

print(f"\n2. Testing with 'a cat':")
try:
    result = generate_2d_image_ai("a cat", 256, 256)
    if isinstance(result, tuple):
        img, method = result
        print(f"   [SUCCESS] Generated using: {method}")
        print(f"   Image size: {img.size}")
    else:
        print(f"   [SUCCESS] Generated (method unknown)")
        print(f"   Image size: {result.size}")
except Exception as e:
    print(f"   [FAILED] Error: {str(e)}")
    import traceback
    print("\n   Full traceback:")
    for line in traceback.format_exc().split('\n'):
        if line.strip():
            print(f"   {line}")

print("\n" + "=" * 60)

