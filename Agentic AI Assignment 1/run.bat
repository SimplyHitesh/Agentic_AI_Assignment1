@echo off
title Statement-to-Insights Finance Agent
echo ===================================================
echo Starting Statement-to-Insights Finance Agent...
echo ===================================================

set PYTHON="C:\Users\hites\AppData\Local\Programs\Python\Python311\python.exe"
set STREAMLIT="C:\Users\hites\AppData\Local\Programs\Python\Python311\Scripts\streamlit.exe"

if exist %STREAMLIT% (
    %STREAMLIT% run app.py
) else (
    %PYTHON% -m streamlit run app.py
)

pause

