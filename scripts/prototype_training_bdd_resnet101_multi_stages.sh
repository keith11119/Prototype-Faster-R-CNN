CUDA_VISIBLE_DEVICES=0,1 python3 fsod_train_net.py --num-gpus 2 --dist-url auto \
	--config-file configs/fsod/prototype_training_bdd_resnet101_stage_1.yaml 2>&1 | tee log/prototype_training_bdd_resnet101_stage_1.txt

CUDA_VISIBLE_DEVICES=0,1 python3 fsod_train_net.py --num-gpus 2 --dist-url auto \
        --config-file configs/fsod/prototype_training_bdd_resnet101_stage_2.yaml 2>&1 | tee log/prototype_training_bdd_resnet101_stage_2.txt

CUDA_VISIBLE_DEVICES=0,1 python3 fsod_train_net.py --num-gpus 2 --dist-url auto \
        --config-file configs/fsod/prototype_training_bdd_resnet101_stage_3.yaml 2>&1 | tee log/prototype_training_bdd_resnet101_stage_3.txt
