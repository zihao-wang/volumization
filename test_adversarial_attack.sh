func() {
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-4 --task_id=wd5e-4 &
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --v=0.25 --alpha=0.5 --task_id=vol
}

func CIFAR100 ResNet18
