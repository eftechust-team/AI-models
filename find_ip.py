#!/usr/bin/env python3
"""
Helper script to find all network IP addresses on your machine.
Run this to see which IP address you should use to access the Flask app.
"""

import socket
import subprocess
import sys

def get_all_ips():
    """Get all IP addresses on all network interfaces"""
    ips = []
    
    # Method 1: Using socket (same as app.py)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        ips.append(("Socket method (default)", ip))
    except Exception as e:
        print(f"Socket method failed: {e}")
    
    # Method 2: Using hostname
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        ips.append(("Hostname method", local_ip))
    except Exception as e:
        print(f"Hostname method failed: {e}")
    
    # Method 3: Using ipconfig (Windows)
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True, shell=True)
        output = result.stdout
        
        # Parse IPv4 addresses from ipconfig output
        for line in output.split('\n'):
            if 'IPv4 Address' in line or 'IPv4 地址' in line:
                ip = line.split(':')[-1].strip()
                if ip and ip != '127.0.0.1':
                    ips.append(("ipconfig", ip))
    except Exception as e:
        print(f"ipconfig method failed: {e}")
    
    return ips

def main():
    print("="*70)
    print("Finding all network IP addresses...")
    print("="*70)
    
    ips = get_all_ips()
    
    if not ips:
        print("No IP addresses found!")
        return
    
    print("\nFound IP addresses:")
    print("-"*70)
    for method, ip in ips:
        print(f"{method:25} -> {ip}")
        print(f"  Try accessing: http://{ip}:5000")
        print()
    
    print("="*70)
    print("Troubleshooting tips:")
    print("1. Make sure Flask app is running (python app.py)")
    print("2. Try each IP address above in your browser")
    print("3. Check Windows Firewall - allow port 5000")
    print("4. Make sure you're on the same network as the server")
    print("5. Try 'localhost' or '127.0.0.1' if accessing from same machine")
    print("="*70)
    
    # Show Windows-specific commands
    print("\nWindows commands to find IP:")
    print("  ipconfig                    - Show all network config")
    print("  ipconfig | findstr IPv4    - Show only IPv4 addresses")
    print("="*70)

if __name__ == "__main__":
    main()

