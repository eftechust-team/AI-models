@echo off
REM Add Windows Firewall rule to allow Flask on port 5000
REM This script must be run as Administrator

echo ======================================================================
echo Adding Windows Firewall rule for Flask (port 5000)
echo ======================================================================
echo.

netsh advfirewall firewall add rule name="Flask Development Server" dir=in action=allow protocol=TCP localport=5000

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Firewall rule added!
    echo Flask server should now be accessible from other devices on your network.
    echo.
) else (
    echo.
    echo [ERROR] Failed to add firewall rule.
    echo Make sure you're running this as Administrator (Right-click -^> Run as administrator)
    echo.
)

echo ======================================================================
pause

