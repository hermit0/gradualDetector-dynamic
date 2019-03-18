log_dir=results_talnn

#从头开始训练模型或者从最近的检查点恢复训练
nohup python -u  main.py \
--root_dir ~/gradualDetector-dynamic/data \
--train_gts_json train+only_gradual.json \
--train_list_path train_samples_c2 \
--result_path $log_dir \
--sample_size 112 \
--sample_duration 21 \
--batch_size 100 \
--n_epochs 50 \
--auto_resume \
--train_subdir train \
--model talnn \
--model_depth 50 \
--n_threads 20 \
--learning_rate 0.01 \
--lr_step 5 \
--checkpoint 1 2>error.log |tee  data/$log_dir/screen.log & 
