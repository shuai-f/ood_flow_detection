param=$1

if [ $param = 1 ];then
    /Users/shuaif/miniforge3/envs/py38/bin/python /Users/shuaif/PycharmProjects/ood_flow_detection/contrast_learning/main.py --epoch 20 --loss sup_nt_xent --n_data_train 260000 --write_summary --draw_figures --temperature 0.1 --base_temperature 0.07

elif [ $param = 2 ];then
    /Users/shuaif/miniforge3/envs/py38/bin/python /Users/shuaif/PycharmProjects/ood_flow_detection/contrast_learning/main.py --epoch 20 --loss npairs --n_data_train 260000 --write_summary --draw_figures

elif [ $param = 3 ];then
    loss=triplet-semihard
    /Users/shuaif/miniforge3/envs/py38/bin/python /Users/shuaif/PycharmProjects/ood_flow_detection/contrast_learning/main.py --epoch 20 --loss $loss --n_data_train 260000 --write_summary --draw_figures
fi
