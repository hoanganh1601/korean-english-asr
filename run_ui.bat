@echo off
chcp 65001 > nul
echo =======================================================
echo  Korean-English ASR Comparator Studio UI
echo =======================================================
echo.
.\.venv\Scripts\python.exe app.py --port 7860
pause
