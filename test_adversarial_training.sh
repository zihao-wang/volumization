adversarial_attack_CIFAR() {
  python eval_adversarial_attack.py --adversarial_training=True \
                                    --dataset=CIFAR10 \
                                    --model=ResNet18 \
                                    --optimizer=vadamw\
                                    "$@"
}

adversarial_attack_MNIST() {
  python eval_adversarial_attack.py --adversarial_training=True \
                                    --dataset=MNIST \
                                    --model=DNN \
                                    --optimizer=vadamw\
                                    "$@"
}


f1(){
  adversarial_attack_CIFAR --case_name="CIFAR10-fgsm-Adv+WD5e-3" --weight_decay=5e-3 --attack_type=fgsm --cuda=2 &
  adversarial_attack_CIFAR --case_name="CIFAR10-fgsm-Adv+WD5e-4" --weight_decay=5e-4 --attack_type=fgsm --cuda=2
  adversarial_attack_CIFAR --case_name="CIFAR10-fgsm-Adv+WD5e-5" --weight_decay=5e-5 --attack_type=fgsm --cuda=2 &
  adversarial_attack_CIFAR --case_name="CIFAR10-fgsm-Adv+Vol(0.125, 0.5)" --v=0.125 --alpha=0.5 --attack_type=fgsm --cuda=2
  adversarial_attack_CIFAR --case_name="CIFAR10-fgsm-Adv+Vol(0.25, -1)" --v=0.25 --alpha=-1 --attack_type=fgsm --cuda=2
}

f2(){
  adversarial_attack_CIFAR --case_name="CIFAR10-pgd20-Adv+WD5e-3" --weight_decay=5e-3 --attack_type=pgd20 --cuda=3 &
  adversarial_attack_CIFAR --case_name="CIFAR10-pgd20-Adv+WD5e-4" --weight_decay=5e-4 --attack_type=pgd20 --cuda=3
  adversarial_attack_CIFAR --case_name="CIFAR10-pgd20-Adv+WD5e-5" --weight_decay=5e-5 --attack_type=pgd20 --cuda=3 &
  adversarial_attack_CIFAR --case_name="CIFAR10-pgd20-Adv+Vol(0.125, 0.5)" --v=0.125 --alpha=0.5 --attack_type=pgd20 --cuda=3
  adversarial_attack_CIFAR --case_name="CIFAR10-pgd20-Adv+Vol(0.25, -1)" --v=0.25 --alpha=-1 --attack_type=pgd20 --cuda=3
}

f3(){
  adversarial_attack_MNIST --case_name="MNIST-fgsm-Adv+WD5e-3" --weight_decay=5e-3 --attack_type=fgsm --cuda=4 &
  adversarial_attack_MNIST --case_name="MNIST-fgsm-Adv+WD5e-4" --weight_decay=5e-4 --attack_type=fgsm --cuda=4
  adversarial_attack_MNIST --case_name="MNIST-fgsm-Adv+WD5e-5" --weight_decay=5e-5 --attack_type=fgsm --cuda=4 &
  adversarial_attack_MNIST --case_name="MNIST-fgsm-Adv+Vol(0.25, 0.5)" --v=0.25 --alpha=0.5 --attack_type=fgsm --cuda=4
  adversarial_attack_MNIST --case_name="MNIST-fgsm-Adv+Vol(0.5, 0.99)" --v=0.5 --alpha=0.99 --attack_type=fgsm --cuda=4
}

f4(){
  adversarial_attack_MNIST --case_name="MNIST-pgd20-Adv+WD5e-3" --weight_decay=5e-3 --attack_type=pgd20 --cuda=5 &
  adversarial_attack_MNIST --case_name="MNIST-pgd20-Adv+WD5e-4" --weight_decay=5e-4 --attack_type=pgd20 --cuda=5
  adversarial_attack_MNIST --case_name="MNIST-pgd20-Adv+WD5e-5" --weight_decay=5e-5 --attack_type=pgd20 --cuda=5 &
  adversarial_attack_MNIST --case_name="MNIST-pgd20-Adv+Vol(0.25, 0.5)" --v=0.25 --alpha=0.5 --attack_type=pgd20 --cuda=5
  adversarial_attack_MNIST --case_name="MNIST-pgd20-Adv+Vol(0.5, 0.99)" --v=0.5 --alpha=0.99 --attack_type=pgd20 --cuda=5
}

f1 &
f2 &
f3 &
f4 &