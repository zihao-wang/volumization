iter_func() {
  for nr in 0.0 0.1 0.2 0.3 0.4
  do
    for alpha in 1 0.9999 0.99 0.5 0 -0.5 -1
    do
      echo CASE: NR=$nr V=$1 alpha=$alpha
      python eval_vol.py --dataset=IMDB --model=LSTMATT --noise_ratio=$nr --alpha=$alpha --task_id=forpub --v=$1 --cuda=$2
    done
  done
}

iter_func 0.25 0 > IMDB-v0.25.log &
iter_func 0.50 1 > IMDB-v0.50.log &
iter_func 1.00 2 > IMDB-v1.00.log &
iter_func 2.00 3 > IMDB-v2.00.log &
iter_func 4.00 4 > IMDB-v4.00.log &

