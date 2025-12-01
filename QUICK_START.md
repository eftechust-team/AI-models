# Quick Start - Getting AI Image Generation Working

## Current Status

OpenAI DALL-E is **not available in your region**. The app will use alternative services.

## Working Solution

The app now tries these services in order:

1. **Replicate API** (if API token set) - Works globally, free tier available
2. **OpenAI DALL-E 2** (if API key set) - May work in some regions
3. **Hugging Face** (free, no key needed) - Currently updating endpoint format

## Quick Setup for Replicate (Recommended)

1. Sign up at https://replicate.com (free account)
2. Get your API token from https://replicate.com/account/api-tokens
3. Set it as environment variable:
   ```powershell
   $env:REPLICATE_API_TOKEN="your_token_here"
   ```
4. Install replicate:
   ```powershell
   pip install replicate
   ```
5. Restart your Flask server

## Testing

Run the app and try generating "a cat" or "a steak". Check the console output to see which service is being used.

If you see "[SUCCESS]" messages, AI generation is working!

