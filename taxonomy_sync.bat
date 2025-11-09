@echo off
REM --- SET PROJECT ROOT DIRECTORY ---
SET PROJECT_ROOT=C:\ProjectK\MarketingAdvantage_AI_v1

REM --- PATHS ---
SET PYTHON_EXE=C:\ProjectK\python.exe
SET SCRIPT_PATH=%PROJECT_ROOT%\api\services\taxonomy_sync.py
SET LOG_FILE=%PROJECT_ROOT%\logs\taxonomy_sync.log

REM --- EXECUTE THE PYTHON SCRIPT ---
echo Running Taxonomy Sync at %DATE% %TIME% >> "%LOG_FILE%"
"%PYTHON_EXE%" -m "%SCRIPT_PATH%" >> "%LOG_FILE%" 2>&1

REM --- CHECK FOR ERRORS (Optional) ---
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: The taxonomy sync script failed. Check the log file for details.
    echo Log file: "%LOG_FILE%"
    goto :end
)

echo.
echo Taxonomy Sync finished successfully. Check the log file for output.

:end
pause