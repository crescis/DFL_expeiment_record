@echo off
:: 启用延迟变量扩展，这是在循环内部正确处理变量的关键
setlocal enabledelayedexpansion

:: 切换到批处理文件所在的目录，确保路径正确
cd /d "%~dp0"

:: ==================================================================
::                     最终版超参数搜索配置
:: ==================================================================
::  说明：请在这里修改您想测试的参数值，用空格隔开。

:: 1) 指定您的 Python 解释器路径
set "PY=C:\Users\Kong\.conda\envs\srdfl\python.exe"

:: 2) 固定参数
set "SPLIT_MODE=iid"
set "CLIENT_NUM=10"
set "EPOCHS=60"
set "GAMMA=0"
set "USE_FEDBN=false"
set "COMMUNICATION_ROUNDS=1 1 1 1 1 1 1 1 1 1"
set "LR_DECAY_STEP=1"

:: 3) 要循环搜索的参数范围
set "BATCH_SELECT_LIST=random periodic"
set "LR_LIST=0.005 0.01 0.05"
set "LOCAL_EPOCHS_LIST=1 2 5 10"
set "BS_LIST=32 64 128 256"
set "LR_DECAY_BETA_LIST=0.980 0.999"

:: ==================================================================
::                     脚本执行部分 (通常无需修改)
:: ==================================================================

:: 基本检查
if not exist "%PY%" (
    echo [ERROR] Python 解释器未找到: %PY%
    pause
    exit /b 1
)
if not exist "train\train_horizontal.py" (
    echo [ERROR] 训练脚本 train\train_horizontal.py 未找到.
    pause
    exit /b 2
)

:: 创建目录
if not exist "logs" mkdir "logs"
if not exist "res" mkdir "res"

echo Starting final grid search...
echo [WARNING] This will run a very large number of experiments.
echo Press Ctrl+C to abort, or any other key to continue...
pause >nul

:: 五层嵌套循环，执行所有参数组合
FOR %%A IN (%BATCH_SELECT_LIST%) DO (
    FOR %%L IN (%LR_LIST%) DO (
        FOR %%E IN (%LOCAL_EPOCHS_LIST%) DO (
            FOR %%B IN (%BS_LIST%) DO (
                FOR %%D IN (%LR_DECAY_BETA_LIST%) DO (
                    :: 为本次运行生成一个唯一的、包含所有参数的标识符
                    set "PARAMS=BSel_%%A-LR_%%L-LE_%%E-BS_%%B-LDB_%%D"
                    set "LOG_FILE=logs\log_!PARAMS!.log"
                    
                    echo.
                    echo ====================================================================
                    echo RUNNING with params: !PARAMS!
                    echo ====================================================================

                    :: 构建并执行完整的 Python 命令
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
                    
                    :: 检查Python脚本的退出代码，判断上一次运行是否成功
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