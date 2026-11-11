@echo off
title Push Project to GitHub
echo ===================================================
echo Pushing Agentic AI Assignment 1 to GitHub...
echo Repository: https://github.com/SimplyHitesh/Agentic_AI_Assignment1.git
echo ===================================================

set GIT="C:\Users\hites\AppData\Local\Programs\MinGit\cmd\git.exe"

if not exist %GIT% (
    set GIT=git
)

%GIT% push -u origin main

echo.
echo ===================================================
echo If prompted, complete the sign-in in your browser.
echo Once done, check: https://github.com/SimplyHitesh/Agentic_AI_Assignment1
echo ===================================================
pause

