"""
Network diagnostic script to help troubleshoot Flask server access
"""
import socket
import subprocess
import sys
import platform

def get_local_ip():
    """Get the local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return None

def get_all_ips():
    """Get all network interface IPs"""
    ips = []
    try:
        if platform.system() == "Windows":
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'IPv4 Address' in line or 'IPv4 地址' in line:
                    ip = line.split(':')[-1].strip()
                    if ip and ip != '127.0.0.1':
                        ips.append(ip)
        else:
            result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
            ips = result.stdout.strip().split()
    except:
        pass
    return ips

def check_port_open(ip, port):
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex((ip, port))
        sock.close()
        return result == 0
    except:
        return False

print("="*70)
print("Network Diagnostic Tool")
print("="*70)
print()

# Get IP addresses
local_ip = get_local_ip()
all_ips = get_all_ips()

print("Your IP Addresses:")
if local_ip:
    print(f"  Primary: {local_ip}")
if all_ips:
    print(f"  All interfaces: {', '.join(all_ips)}")
print()

# Check if Flask is running
print("Checking if Flask server is running on port 5000...")
if local_ip:
    if check_port_open(local_ip, 5000):
        print(f"  ✓ Port 5000 is OPEN on {local_ip}")
        print(f"  ✓ Server should be accessible at: http://{local_ip}:5000")
    else:
        print(f"  ✗ Port 5000 is CLOSED on {local_ip}")
        print("  → Make sure Flask server is running (python app.py)")
        print("  → Check Windows Firewall settings")
else:
    print("  ✗ Could not determine IP address")

print()
print("="*70)
print("Troubleshooting Steps:")
print("="*70)
print("1. Make sure Flask server is running: python app.py")
print("2. Check Windows Firewall:")
print("   - Open Windows Defender Firewall")
print("   - Click 'Advanced settings'")
print("   - Add inbound rule for port 5000 (TCP)")
print("3. Verify same network:")
print("   - All devices must be on the same Wi-Fi/network")
print("   - Check IP addresses match network (e.g., 192.168.1.x)")
print("4. Test from another device:")
if local_ip:
    print(f"   - Try: http://{local_ip}:5000")
print("5. If still not working, temporarily disable firewall to test")
print("="*70)

