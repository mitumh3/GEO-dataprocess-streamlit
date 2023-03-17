@echo off
call env\Scripts\activate.bat
pip install -r requirements.txt

echo Done.

call env\Scripts\deactivate.bat
echo Done > "exist.txt"
cd ..\
call run.bat