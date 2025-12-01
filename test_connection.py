#!/usr/bin/env python3
"""
Test script to verify Flask server connectivity
"""

import socket
import requests
import sys

def test_port(host, port):
    """Test if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception as e:
        print(f"Error testing {host}:{port}: {e}")
        return False

def test_http(host, port):
    """Test HTTP connection"""
    try:
        url = f"http://{host}:{port}"
        response = requests.get(url, timeout=5)
        return True, response.status_code
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except requests.exceptions.Timeout:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    print("="*70)
    print("Testing Flask Server Connectivity")
    print("="*70)
    
    # Get all IPs
    import subprocess
    result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
    ips = []
    for line in result.stdout.split('\n'):
        if 'IPv4 Address' in line or 'IPv4 地址' in line:
            ip = line.split(':')[-1].strip()
            if ip and ip != '127.0.0.1':
                ips.append(ip)
    
    # Test localhost first
    print("\n1. Testing localhost:5000...")
    port_open = test_port('127.0.0.1', 5000)
    print(f"   Port 5000 open: {port_open}")
    
    if port_open:
        http_ok, status = test_http('127.0.0.1', 5000)
        print(f"   HTTP connection: {http_ok} (Status: {status})")
    else:
        print("   ❌ Port 5000 is not open on localhost!")
        print("   Make sure Flask app is running: python app.py")
        return
    
    # Test all IP addresses
    print("\n2. Testing network IP addresses...")
    for ip in ips:
        print(f"\n   Testing {ip}:5000...")
        port_open = test_port(ip, 5000)
        print(f"   Port open: {port_open}")
        
        if port_open:
            http_ok, status = test_http(ip, 5000)
            print(f"   HTTP connection: {http_ok} (Status: {status})")
            if http_ok:
                print(f"   [SUCCESS] Use: http://{ip}:5000")
            else:
                print(f"   [WARNING] Port open but HTTP failed: {status}")
        else:
            print(f"   ❌ Port not accessible from network")
    
    print("\n" + "="*70)
    print("Troubleshooting:")
    print("="*70)
    print("If localhost works but network IPs don't:")
    print("1. Windows Firewall is likely blocking - add exception for port 5000")
    print("2. Run as Administrator: netsh advfirewall firewall add rule name='Flask' dir=in action=allow protocol=TCP localport=5000")
    print("3. Or temporarily disable firewall to test")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
