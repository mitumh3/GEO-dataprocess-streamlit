@echo off

REM Check if pip is installed
pip --version > nul 2>&1
if %errorlevel% equ 0 (
    echo Pip is already installed.
) else (
    echo Installing pip...
    REM Download get-pip.py
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    
    REM Install pip
    python get-pip.py
    
    REM Clean up
    del get-pip.py
)
timeout /t 5 >nul
call virtualenv_install.bat