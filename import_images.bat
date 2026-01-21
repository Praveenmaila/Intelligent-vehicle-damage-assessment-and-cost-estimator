@echo off
REM Quick launcher for importing vehicle damage images

echo ========================================
echo   Vehicle Damage Image Importer
echo ========================================
echo.
echo This will import your 1500+ images into the dataset.
echo.
pause

python import_images.py

echo.
echo ========================================
echo   Import Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Review: datasets\data.csv
echo   2. Train model: python train_improved.py
echo   3. Run app: python app.py
echo.
pause
