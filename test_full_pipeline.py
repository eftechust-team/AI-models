"""
Test the full pipeline: text -> AI image -> silhouette -> contour -> STL
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import generate_2d_image, image_to_contour, extrude_2d_to_3d

print("="*60)
print("Testing Full Pipeline")
print("="*60)

test_prompts = [
    "a cat",
    "a steak", 
    "a circle"
]

for prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"Testing: '{prompt}'")
    print(f"{'='*60}")
    
    try:
        # Step 1: Generate 2D image
        print(f"\nStep 1: Generating 2D image...")
        img = generate_2d_image(prompt, 512, 512)
        print(f"[OK] Image generated, size: {img.size}")
        
        # Step 2: Extract contour
        print(f"\nStep 2: Extracting contour...")
        contour = image_to_contour(img)
        if contour is None:
            print(f"[ERROR] No contour found!")
            continue
        print(f"[OK] Contour extracted, points: {len(contour)}")
        
        # Step 3: Extrude to 3D
        print(f"\nStep 3: Extruding to 3D...")
        thickness = 10.0
        mesh_3d = extrude_2d_to_3d(contour, thickness)
        print(f"[OK] 3D mesh created")
        
        # Step 4: Check mesh bounds
        min_bounds = mesh_3d.vectors.min(axis=(0, 1))
        max_bounds = mesh_3d.vectors.max(axis=(0, 1))
        dimensions = max_bounds - min_bounds
        print(f"[OK] Mesh dimensions: {dimensions[0]:.2f} x {dimensions[1]:.2f} x {dimensions[2]:.2f} mm")
        
        print(f"\n[SUCCESS] Pipeline works for '{prompt}'!")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] Failed for '{prompt}': {str(e)}")
        print(f"Traceback:")
        traceback.print_exc()
        continue

print(f"\n{'='*60}")
print("Test Complete")
print("="*60)

