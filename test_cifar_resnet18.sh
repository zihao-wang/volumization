for nr in 0.0 0.2 0.5 0.8
do
  for v in 0.1 0.2 0.3 0.5
  do
    echo CASE: NR $nr V $v
    python eval_cifar.py --noise_ratio=$nr --v=$v --task_id=try
  done
done