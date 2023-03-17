@echo off
call etwenv\Scripts\activate.bat
pip install -r requirements.txt

echo Done.

call etwenv\Scripts\deactivate.bat
echo Done > "exist.txt"
cd ..\
call run.bat