@echo off
if not exist ".\scripts\setup-done.txt" (
    cd script
    call python_install.bat
) else (
    cd script
    call env\Scripts\activate.bat
    streamlit run app_final.py
)