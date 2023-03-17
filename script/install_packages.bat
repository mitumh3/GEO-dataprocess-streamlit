@echo off
call env\Scripts\activate.bat
pip install -r requirements.txt

echo Done.

call env\Scripts\deactivate.bat
echo Done > "setup-done.txt"
cd ..\
cls
call run.bat