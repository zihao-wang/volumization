for nr in 0.0 0.2 0.4 0.6 0.8
do
  for alpha in -0.99 -0.9
  do
    echo CASE: NR=$nr V=$2 alpha=$alpha
    python eval_mnist.py --noise_ratio=$nr --alpha=$alpha --task_id=clarify_v_alpha --cuda=$1 --v=$2
  done
done
