# Vercel Deployment Size Fix

## Problem
Vercel serverless functions have a **250 MB unzipped size limit**. Your deployment is exceeding this limit.

## Root Cause
Even with minimal requirements, Python packages can be large:
- Flask + dependencies: ~30-50 MB
- flask-cors: ~5 MB  
- requests: ~10-15 MB
- Total with all dependencies: Can exceed 250 MB when bundled

## Solutions Applied

### 1. ✅ Created `.vercelignore`
Excludes unnecessary files from deployment:
- Test files
- Documentation
- Saved models
- Build artifacts
- IDE files

### 2. ✅ Optimized `requirements-vercel-minimal.txt`
Removed heavy packages:
- ❌ Pillow (~50 MB) - removed
- ❌ numpy, scipy, scikit-image, scikit-learn - removed
- ✅ Only Flask, flask-cors, requests - kept

### 3. ✅ Updated `vercel.json`
- Uses minimal requirements
- Optimized install command
- Added function configuration

## Current Status

The deployment should now be under 250 MB, but **some features will be limited**:
- ✅ Basic Flask app works
- ✅ API endpoints work
- ❌ Image processing (requires Pillow)
- ❌ STL generation (requires numpy-stl)

## If Still Exceeding 250 MB

### Option 1: Ultra-Minimal Requirements
Use only Flask (remove flask-cors if CORS not needed):

```txt
Flask>=3.0.0
```

Then handle CORS manually in code if needed.

### Option 2: Use Vercel Output File System
Move static files to Vercel's output file system to reduce function size.

### Option 3: Split into Multiple Functions
- Create separate API routes as separate functions
- Each function stays under 250 MB
- Use Vercel's routing to connect them

### Option 4: Use Different Platform (Recommended for Full Features)
For full functionality with all dependencies:
- **Railway** - No size limits, better for Flask apps
- **Render** - Supports larger functions (512 MB)
- **Fly.io** - Container-based, full control
- **Heroku** - Classic option (paid)

### Option 5: External Services
- Move image processing to external API
- Move STL generation to separate service
- Keep main function minimal

## Testing the Fix

1. **Commit the changes:**
   ```bash
   git add .vercelignore requirements-vercel-minimal.txt vercel.json
   git commit -m "Fix Vercel deployment size limit"
   git push
   ```

2. **Redeploy on Vercel**
   - The build should now use minimal requirements
   - Check build logs for size

3. **Verify function size:**
   - Check Vercel dashboard
   - Function should be under 250 MB

## Next Steps

If deployment succeeds but features are missing:
1. Add back packages one at a time
2. Monitor size after each addition
3. Consider moving heavy operations to external services

## Alternative: Minimal API-Only Deployment

If you only need API endpoints, create a minimal `api/index.py`:

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({"message": "API is running"})

# Export for Vercel
handler = app
```

This keeps the function extremely small.

