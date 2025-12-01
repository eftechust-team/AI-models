# Fixing Flask Server Network Access

## Problem
The Flask server works on `localhost` but cannot be accessed from other devices using your network IP address.

## Solution: Add Windows Firewall Rule

### Option 1: Run the batch script (Easiest)
1. **Right-click** on `setup_firewall.bat`
2. Select **"Run as administrator"**
3. The script will add a firewall rule automatically

### Option 2: Manual PowerShell command (Run as Admin)
Open PowerShell as Administrator and run:
```powershell
netsh advfirewall firewall add rule name="Flask Development Server" dir=in action=allow protocol=TCP localport=5000
```

### Option 3: Windows Firewall GUI
1. Open **Windows Defender Firewall**
2. Click **"Advanced settings"**
3. Click **"Inbound Rules"** → **"New Rule"**
4. Select **"Port"** → Next
5. Select **"TCP"** and enter port **5000** → Next
6. Select **"Allow the connection"** → Next
7. Check all profiles (Domain, Private, Public) → Next
8. Name it "Flask Development Server" → Finish

## Verify Your IP Address

Run this command to see your current IP:
```powershell
ipconfig | findstr IPv4
```

Or run:
```powershell
python find_ip.py
```

## Test Connection

After adding the firewall rule:
1. Make sure Flask is running: `python app.py`
2. Try accessing from another device: `http://YOUR_IP:5000`
3. Or test locally: `python test_connection.py`

## Common Issues

1. **Still can't access?**
   - Make sure both devices are on the same Wi-Fi network
   - Try temporarily disabling firewall to test
   - Check if your router has AP isolation enabled (disable it)

2. **IP address changed?**
   - IP addresses can change when you reconnect to Wi-Fi
   - Run `python find_ip.py` again to get the new IP

3. **Port already in use?**
   - Another program might be using port 5000
   - Change the port in `app.py`: `app.run(host='0.0.0.0', debug=True, port=8080)`

