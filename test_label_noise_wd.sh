iter_func() {
for wd in 5e-3 5e-4 5e-5
  do
    python eval_vol.py --dataset=$1 --model=$2 --noise_ratio=$3 --weight_decay=$wd --task_id=label_noise$3_wd$wd
  done
}


iter_func MNIST DNN 0.4 &
iter_func MNIST DNN 0.8

iter_func CIFAR10 ResNet18 0.4 &
iter_func CIFAR10 ResNet18 0.8

iter_func IMDB LSTMATT 0.2 &
iter_func IMDB LSTMATT 0.4