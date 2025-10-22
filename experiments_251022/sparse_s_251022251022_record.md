在sparse tops的设定下展开超参数搜索，通过expeiment_0918.bat实现，
参数搜索范围包括：
BATCH_SELECT_LIST: random, periodic
LR_LIST: 0.005,0.01,0.05
LOCAL_EPOCHS_LIST: 1,2,5,10
BS_LIST: 32,64,128,256
LR_DECAY_BETA_LIST: 0.980,0.999
结果：
BATCH_SELECT_LIST：random略优，在所有场景下acc平均优于periodic 0.3% 左右
<img width="1800" height="1400" alt="strategy_delta_relationships" src="https://github.com/user-attachments/assets/8948a804-ba9f-438d-9597-d3f2631c0e07" />

LR_LIST: 总体而言lr=0.05时acc更好
<img width="1800" height="1400" alt="lr_hyperparameter_relationships" src="https://github.com/user-attachments/assets/d50546f3-25fc-4ba0-bee2-d7552ce4d3ac" />

LOCAL_EPOCHS_LIST: 总体而言local_epoches=5时acc更好
<img width="1800" height="1400" alt="local_epochs_hyperparameter_relationships" src="https://github.com/user-attachments/assets/abff16e4-2bbf-4f8a-845b-3b59d25d8f51" />

BS_LIST: bs=32时效果更好
<img width="1800" height="1400" alt="bs_hyperparameter_relationships" src="https://github.com/user-attachments/assets/4d020453-28f2-469c-a79d-6c36d048e0f4" />

LR_DECAY_BETA_LIST: 差别较小，目前0.98时更好
<img width="1800" height="1400" alt="lr_decay_beta_hyperparameter_relationships" src="https://github.com/user-attachments/assets/a9724d65-5d3b-471d-8897-3b6c9898adde" />

Heatmap:
<img width="2000" height="1400" alt="heatmap_strategy_pairs" src="https://github.com/user-attachments/assets/fda27e19-8de0-4d1c-81c5-07ef11d26a6d" />

总的来说，lr,local_epochs_bs呈现了明显趋势，在所有场景下存在一个最优解，batch_select和lr_decay_beta趋势相对不是很明显
