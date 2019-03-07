log_dir=results_temp

#根据训练好的模型文件来在测试集上进行测试
nohup python -u main.py \
--no_train \
--test \
--root_dir ~/gradualDetector/data \
--test_list_path temp_list \
--result_path $log_dir \
--sample_size 112 \
--sample_duration 21 \
--batch_size 64 \
--resume_path $log_dir/model_epoch4.pth \
--test_subdir train \
--model talnn \
--model_depth 50 \
--n_threads 20 |tee  data/$log_dir/screen_test.log &
