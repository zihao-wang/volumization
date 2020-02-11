func() {
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --task_id=plain --cuda=0 --case_name="No Regularization" --load=True &
#  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-1 --task_id=wd --optimizer=vadamw --cuda=0 --case_name=WD5e-1 --load=True&
#  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-2 --task_id=wd --optimizer=vadamw --cuda=1 --case_name=WD5e-2 --load=True&
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-3 --task_id=wd --optimizer=vadamw --cuda=1 --case_name=WD5e-3 --load=True&
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-4 --task_id=wd --optimizer=vadamw --cuda=2 --case_name=WD5e-4 --load=True&
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --weight_decay=5e-5 --task_id=wd --optimizer=vadamw --cuda=2 --case_name=WD5e-5 --load=True&
#  python eval_adversarial_attack.py --dataset=$1 --model=$2 --v=0.25 --alpha=0.5 --task_id=v0_25 --optimizer=vadamw --cuda=3 --case_name="Vol(0.25, 0.5)" --load=True&
#  python eval_adversarial_attack.py --dataset=$1 --model=$2 --v=0.5 --alpha=0.99 --task_id=v0_5 --optimizer=vadamw --cuda=3 --case_name="Vol(0.5, 0.99)" --load=False&
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --v=0.25 --alpha=-1 --task_id=v0_25 --optimizer=vadamw --cuda=3 --case_name="Vol(0.25, -1)" --load=True&
  python eval_adversarial_attack.py --dataset=$1 --model=$2 --v=0.125 --alpha=0.5 --task_id=v0_125 --optimizer=vadamw --cuda=3 --case_name="Vol(0.125, 0.5)" --load=True
}

func CIFAR10 ResNet18
#func MNIST DNN
