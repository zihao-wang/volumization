for nr in 0.0 0.2 0.4 0.6 0.8
do
  for v in 1 5 10 50 100
  do
    echo CASE: NR $nr V $v
    python eval_cifar.py --noise_ratio=$nr --v=$v --task_id=try
  done
done