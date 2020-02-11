CIFAR10_addv() {
for v in 0.25 0.5 1.0 2.0
do
  python eval_vol.py --dataset=CIFAR10 --model=ResNet18 --noise_ratio=0.4 --v=$v --alpha=-1  --task_id="add_0.4_v_alpha=-1"  --cuda=0 &
  python eval_vol.py --dataset=CIFAR10 --model=ResNet18 --noise_ratio=0.8 --v=$v --alpha=0.5 --task_id="add_0.8_v_alpha=0_5" --cuda=1
done
}

CIFAR10_addalpha() {
for a in -1.0 -0.5 0 0.5 0.99 0.9999 1.0
do
  python eval_vol.py --dataset=CIFAR10 --model=ResNet18 --noise_ratio=0.4 --v=0.25 --alpha=$a --task_id="add_0.4_alpha_v=0.25" --cuda=0 &
  python eval_vol.py --dataset=CIFAR10 --model=ResNet18 --noise_ratio=0.8 --v=0.25 --alpha=$a --task_id="add_0.8_alpha_v=0.25" --cuda=1
done
}

IMDB_addv() {
for v in 0.25 0.5 1.0 2.0
do
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.4 --v=$v --alpha=0.5 --task_id="add_0.4_v_alpha=0_5" --cuda=4 &
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.2 --v=$v --alpha=0.5 --task_id="add_0.2_v_alpha=0_5" --cuda=2 &
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.2 --v=$v --alpha=0.5 --task_id="add_0.2_v_alpha=0_5" --cuda=2
done
}

IMDB_addalpha() {
for a in -1.0 -0.5 0 0.5 0.99 0.9999 1.0
do
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.4 --v=0.5 --alpha=$a --task_id="add_0.4_alpha_v=0.25" --cuda=4 &
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.2 --v=0.5 --alpha=$a --task_id="add_0.2_alpha_v=0.25" --cuda=2 &
  python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=0.2 --v=0.5 --alpha=$a --task_id="add_0.2_alpha_v=0.25" --cuda=2
done
}

IMDB_addv &
IMDB_addalpha &
CIFAR10_addv &
CIFAR10_addalpha &
