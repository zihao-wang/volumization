for nr in 0 0.1 0.2 0.3 0.4
do
  for v in 0.1 0.2 0.4 0.6 0.8 1.0
  do
    echo CASE: NR $nr V $v
    python eval_imdb.py --dataset=IMDB --model=LSTM --noise_ratio=$nr --v=$v --task_id=final
  done
done
