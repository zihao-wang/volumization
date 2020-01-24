iter_func() {
  for nr in 0.0 0.2 0.4 0.6 0.8
  do
    for alpha in 1 0.9999 0.99 0.5 0 -0.5 -1
    do
      echo CASE: NR=$nr V=$1 alpha=$alpha
      python eval_vol.py --dataset=CIFAR10 --model=ResNet18 --noise_ratio=$nr --alpha=$alpha --task_id=forpub --v=$1 --cuda=$2
    done
  done
}

iter_func 0.25 0 > cifar10-v0.25.log &
iter_func 0.50 1 > cifar10-v0.50.log &
iter_func 1.00 2 > cifar10-v1.00.log &
iter_func 2.00 3 > cifar10-v2.00.log &

