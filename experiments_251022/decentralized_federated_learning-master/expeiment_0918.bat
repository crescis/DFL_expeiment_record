@echo off
:: �����ӳٱ�����չ��������ѭ���ڲ���ȷ��������Ĺؼ�
setlocal enabledelayedexpansion

:: �л����������ļ����ڵ�Ŀ¼��ȷ��·����ȷ
cd /d "%~dp0"

:: ==================================================================
::                     ���հ泬������������
:: ==================================================================
::  ˵�������������޸�������ԵĲ���ֵ���ÿո������

:: 1) ָ������ Python ������·��
set "PY=C:\Users\Kong\.conda\envs\srdfl\python.exe"

:: 2) �̶�����
set "SPLIT_MODE=iid"
set "CLIENT_NUM=10"
set "EPOCHS=60"
set "GAMMA=0"
set "USE_FEDBN=false"
set "COMMUNICATION_ROUNDS=1 1 1 1 1 1 1 1 1 1"
set "LR_DECAY_STEP=1"

:: 3) Ҫѭ�������Ĳ�����Χ
set "BATCH_SELECT_LIST=random periodic"
set "LR_LIST=0.005 0.01 0.05"
set "LOCAL_EPOCHS_LIST=1 2 5 10"
set "BS_LIST=32 64 128 256"
set "LR_DECAY_BETA_LIST=0.980 0.999"

:: ==================================================================
::                     �ű�ִ�в��� (ͨ�������޸�)
:: ==================================================================

:: �������
if not exist "%PY%" (
    echo [ERROR] Python ������δ�ҵ�: %PY%
    pause
    exit /b 1
)
if not exist "train\train_horizontal.py" (
    echo [ERROR] ѵ���ű� train\train_horizontal.py δ�ҵ�.
    pause
    exit /b 2
)

:: ����Ŀ¼
if not exist "logs" mkdir "logs"
if not exist "res" mkdir "res"

echo Starting final grid search...
echo [WARNING] This will run a very large number of experiments.
echo Press Ctrl+C to abort, or any other key to continue...
pause >nul

:: ���Ƕ��ѭ����ִ�����в������
FOR %%A IN (%BATCH_SELECT_LIST%) DO (
    FOR %%L IN (%LR_LIST%) DO (
        FOR %%E IN (%LOCAL_EPOCHS_LIST%) DO (
            FOR %%B IN (%BS_LIST%) DO (
                FOR %%D IN (%LR_DECAY_BETA_LIST%) DO (
                    :: Ϊ������������һ��Ψһ�ġ��������в����ı�ʶ��
                    set "PARAMS=BSel_%%A-LR_%%L-LE_%%E-BS_%%B-LDB_%%D"
                    set "LOG_FILE=logs\log_!PARAMS!.log"
                    
                    echo.
                    echo ====================================================================
                    echo RUNNING with params: !PARAMS!
                    echo ====================================================================

                    :: ������ִ�������� Python ����
                    "%PY%" -u -m train.train_horizontal ^
                        --split_mode %SPLIT_MODE% ^
                        --client_num %CLIENT_NUM% ^
                        --epochs %EPOCHS% ^
                        --gamma %GAMMA% ^
                        --use_fedbn %USE_FEDBN% ^
                        --communication_rounds %COMMUNICATION_ROUNDS% ^
                        --lr_decay_step %LR_DECAY_STEP% ^
                        --batch_select %%A ^
                        --lr %%L ^
                        --local_epochs %%E ^
                        --bs %%B ^
                        --lr_decay_beta %%D ^
                        > "!LOG_FILE!" 2>&1
                    
                    :: ���Python�ű����˳����룬�ж���һ�������Ƿ�ɹ�
                    if errorlevel 1 (
                        echo FAILED: !PARAMS!. See log: !LOG_FILE!
                    ) else (
                        echo FINISHED: !PARAMS!. Log saved to: !LOG_FILE!
                    )
                )
            )
        )
    )
)

echo.
echo Grid search has finished.
echo All logs are in the 'logs' directory.
echo All experiment results (JSON files) are in the 'res' directory.
pause
endlocal