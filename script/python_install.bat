@echo off

rem Check if Python is already installed
python --version > nul 2>&1
if %errorlevel% == 0 (
    echo Python is already installed
) else (
    rem Download and install Python
    powershell -Command "Invoke-WebRequest https://www.python.org/ftp/python/3.10.2/python-3.10.2-amd64.exe -OutFile python-installer.exe"
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
    del python-installer.exe
    echo Python has been installed
)
rem --Refresh Environmental Variables
call pip_install.bat