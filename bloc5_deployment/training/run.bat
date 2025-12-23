@echo off

docker run -it ^
 -p 7860:7860 ^
 -v "%cd%\..\data:/home/app/data" ^
 -v "%cd%\..\models:/home/app/models" ^
 -e MLFLOW_TRACKING_URI=%MLFLOW_TRACKING_URI% ^
 -e AWS_ACCESS_KEY_ID=%AWS_ACCESS_KEY_ID% ^
 -e AWS_SECRET_ACCESS_KEY=%AWS_SECRET_ACCESS_KEY% ^
 training-image python train.py --model xgb