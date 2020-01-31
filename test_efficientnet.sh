iter_func() {
 python eval_EfficientNet.py --v=1 --alpha=0.99 --task_id=finetune-volumization --batch_size=$1 --model=$2 --cuda=$3
 python eval_EfficientNet.py --weight_decay=5e-4 --task_id=finetune-weight-decay --batch_size=$1 --model=$2 --cuda=$3
}

iter_func 32 efficientnet-b2 0 > b2.log &
iter_func 32 efficientnet-b3 1 > b3.log &
iter_func 8  efficientnet-b4 2 > b4.log &
iter_func 4  efficientnet-b5 3 > b5.log &
iter_func 4  efficientnet-b6 4 > b6.log &
iter_func 2  efficientnet-b7 5 > b7.log &