@echo off
echo ===================================================
echo   Network Intrusion Detection System (NIDS) Pipeline
echo ===================================================

echo Installing dependencies...
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm flask mlflow kafka-python

echo.
echo ===================================================
echo   STAGE 1: Data Preprocessing
echo ===================================================
cd nids_stage1\src
python pipeline.py --dataset cic_ids2017 --path data\sample_cic_ids2017.csv
python pipeline.py --dataset unsw_nb15 --path data\sample_unsw_nb15.csv
cd ..\..

echo.
echo ===================================================
echo   STAGE 2: Model Training
echo ===================================================
cd nids_stage2\src
python train_stage2.py --dataset cic_ids2017
python train_stage2.py --dataset unsw_nb15
cd ..\..

echo.
echo ===================================================
echo   STAGE 3: Advanced Training (Class Imbalance)
echo ===================================================
cd nids_stage3\src
python train_stage3.py --dataset cic_ids2017
python train_stage3.py --dataset unsw_nb15
cd ..\..

echo.
echo ===================================================
echo   STAGE 4: Explainable AI Models
echo ===================================================
cd nids_stage4\src
python train_stage4.py
cd ..\..

echo.
echo ===================================================
echo   STAGE 5: XAI Dashboard
echo ===================================================
echo Starting XAI Dashboard...
echo Open http://localhost:5000 in your browser.
cd nids_stage5\src\dashboard
start python app.py --dataset cic_ids2017
cd ..\..\..

echo.
echo ===================================================
echo   STAGE 6: Real-time Inference Engine
echo ===================================================
cd "%ROOT_DIR%\nids_stage6"
python -u src\run_realtime_demo.py --mode simulate --stage1_dir "%DATA_DIR%\processed\cic_ids2017"
cd ..\..

echo.
echo Pipeline execution complete.
pause


// PS C:\Users\Hp\Downloads\nids_zipped\nids> cd nids_stage5\src\dashboard
PS C:\Users\Hp\Downloads\nids_zipped\nids\nids_stage5\src\dashboard> python app.py --dataset cic_ids2017 --stage1_dir "C:\Users\Hp\Downloads\nids_zipped\nids\data\processed\cic_ids2017"