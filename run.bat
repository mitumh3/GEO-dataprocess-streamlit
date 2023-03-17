@echo off
if not exist ".\script\exist.txt" (
    cd script
    call python_install.bat
) else (
    cd script
    call etwenv\Scripts\activate.bat
    streamlit run app_final.py
)
