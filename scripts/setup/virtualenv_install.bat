@echo off
:: Check for numpy package
pip show virtualenv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing virtualenv package...
    pip install virtualenv==20.19.0
)
python -m virtualenv env
:: install packages script
call install_packages.bat