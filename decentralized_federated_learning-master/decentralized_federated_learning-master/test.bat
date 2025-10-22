@echo off
:: 启用延迟变量扩展，这是在循环内部正确处理变量的关键
setlocal enabledelayedexpansion

:: 切换到批处理文件所在的目录，确保路径正确
cd /d "%~dp0"

:: ==================================================================
::                     超参数搜索配置
:: ==================================================================
::  说明：请在这里修改您想测试的参数值，用空格隔开。

:: 1) 指定您的 Python 解释器路径
set "PY=C:\Users\Kong\.conda\envs\srdfl\python.exe"

:: 2) 贯穿所有实验的固定参数
set "CLIENT_NUM=10"
set "EPOCHS=50"
set "INITIAL_NEIGHBOR_RATE=1.0"
set "USE_FEDBN=true"
set "COMMUNICATION_PERIOD=1 1 1 1 1 1 1 1 1 1"

:: ------------------- STAGE 1: 核心参数搜索范围 --------------------
set "STAGE1_LR_LIST=0.01 0.005"
set "STAGE1_LE_LIST=1 5 10"
set "STAGE1_SR_LIST=0.3 0.7 1.0"
:: STAGE 1 中固定的参数
set "STAGE1_BS=64"
set "STAGE1_CR=0.8"

:: ------------------- STAGE 2: 次要参数搜索范围 --------------------
set "STAGE2_BS_LIST=32 64 128"
set "STAGE2_CR_LIST=0.4 0.8 1.0"
:: STAGE 2 中固定的参数 (重要：请在跑完第一阶段后，手动填入最佳结果)
set "BEST_LR=0.01"
set "BEST_LE=5"
set "BEST_SR=0.7"

:: ==================================================================
::                     脚本执行部分
:: ==================================================================

:: 基本检查
if not exist "%PY%" (echo [ERROR] Python 解释器未找到: %PY% & pause & exit /b)
if not exist "train\train_horizontal.py" (echo [ERROR] 训练脚本未找到 & pause & exit /b)

:: 创建目录
if not exist "logs" mkdir "logs"
if not exist "res" mkdir "res"

echo Starting hyperparameter sweep...
echo Press Ctrl+C to abort, or any other key to continue...
pause >nul

:: ##################################################################
:: #                       第一阶段：核心参数搜索                     #
:: ##################################################################
echo.
echo ^> ^> ^> ^> ^> ^> ^> ^> ^> ^> STAGE 1: Sweeping Core Parameters (lr, local_epochs, s_ratio) ^< ^< ^< ^< ^< ^< ^< ^< ^< ^<

FOR %%L IN (%STAGE1_LR_LIST%) DO (
    FOR %%E IN (%STAGE1_LE_LIST%) DO (
        FOR %%S IN (%STAGE1_SR_LIST%) DO (
            set "PARAMS=STAGE1_LR%%L-LE%%E-SR%%S"
            set "LOG_FILE=logs\log_!PARAMS!.log"
            
            echo.
            echo --- RUNNING: !PARAMS! ---
            
            "%PY%" -u -m train.train_horizontal ^
                --client_num %CLIENT_NUM% ^
                --epochs %EPOCHS% ^
                --initial_neighbor_rate %INITIAL_NEIGHBOR_RATE% ^
                --use_fedbn %USE_FEDBN% ^
                --communication_period %COMMUNICATION_PERIOD% ^
                --bs %STAGE1_BS% ^
                --communication_rate %STAGE1_CR% ^
                --lr %%L ^
                --local_epochs %%E ^
                --s_ratio %%S ^
                > "!LOG_FILE!" 2>&1
            
            if errorlevel 1 (
                echo FAILED: !PARAMS!. See log: !LOG_FILE!
            ) else (
                echo FINISHED: !PARAMS!. Log saved to: !LOG_FILE!
            )
        )
    )
)

:: ##################################################################
:: #                       第二阶段：次要参数微调                     #
:: #  运行此阶段前，请务必根据第一阶段的结果，修改上面 BEST_LR/LE/SR 的值  #
:: ##################################################################

REM echo.
REM echo ^> ^> ^> ^> ^> ^> ^> ^> ^> ^> STAGE 2: Sweeping Secondary Parameters (bs, communication_rate) ^< ^< ^< ^< ^< ^< ^< ^< ^< ^<
REM 
REM FOR %%B IN (%STAGE2_BS_LIST%) DO (
REM     FOR %%R IN (%STAGE2_CR_LIST%) DO (
REM         set "PARAMS=STAGE2_BS%%B-CR%%R_BEST_LR%BEST_LR%-LE%BEST_LE%-SR%BEST_SR%"
REM         set "LOG_FILE=logs\log_!PARAMS!.log"
REM         
REM         echo.
REM         echo --- RUNNING: !PARAMS! ---
REM         
REM         "%PY%" -u -m train.train_horizontal ^
REM             --client_num %CLIENT_NUM% ^
REM             --epochs %EPOCHS% ^
REM             --initial_neighbor_rate %INITIAL_NEIGHBOR_RATE% ^
REM             --use_fedbn %USE_FEDBN% ^
REM             --communication_period %COMMUNICATION_PERIOD% ^
REM             --lr %BEST_LR% ^
REM             --local_epochs %BEST_LE% ^
REM             --s_ratio %BEST_SR% ^
REM             --bs %%B ^
REM             --communication_rate %%R ^
REM             > "!LOG_FILE!" 2>&1
REM         
REM         if errorlevel 1 (
REM             echo FAILED: !PARAMS!. See log: !LOG_FILE!
REM         ) else (
REM             echo FINISHED: !PARAMS!. Log saved to: !LOG_FILE!
REM         )
REM     )
REM )

echo.
echo Full sweep has finished.
echo All logs are in the 'logs' directory.
echo All experiment results (JSON files) are in the 'res' directory.
pause
endlocal