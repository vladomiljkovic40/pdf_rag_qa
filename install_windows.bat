@echo off
echo  PDF RAG QA System - Windows Installation Script
echo ================================================

echo  Checking Python installation...
py --version
if errorlevel 1 (
    echo  Python not found! Please install Python 3.8+ first.
    echo  Download from: https://python.org
    pause
    exit /b 1
)

echo  Installing required packages...
py -m pip install -r requirements.txt

if errorlevel 1 (
    echo  Installation failed! Please check error messages above.
    pause
    exit /b 1
)

echo  Installation completed successfully!
echo  You can now run: python pdf_rag_qa.py
echo  Or run test: python test_rag_system.py

pause
