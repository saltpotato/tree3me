@echo off
setlocal

REM Activate venv if it exists
if exist myenv\Scripts\activate.bat (
    call myenv\Scripts\activate.bat
)

echo.
echo ================================
echo  TREE3 ML Pipeline
echo ================================
echo.

echo [1/4] Removing old training_data.csv...
if exist training_data.csv del training_data.csv

echo.
echo [2/4] Collecting heuristic training data...
python run_experiment.py --mode collect --episodes 500
if errorlevel 1 goto error

echo.
echo [3/4] Training imitation model...
python train_imitation_model.py
if errorlevel 1 goto error

echo.
echo [4/4] Running benchmark...
python run_experiment.py --mode benchmark --episodes 50
if errorlevel 1 goto error

echo.
echo ================================
echo  Done.
echo ================================
goto end

:error
echo.
echo ERROR: Pipeline failed.
exit /b 1

:end
endlocal
pause