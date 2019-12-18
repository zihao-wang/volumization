for nr in 0.0 0.2 0.5 0.8
do
  for v in 0.1 0.2 0.5
  do
    echo CASE: NR $nr V $v
    python eval.py --dataset=IMDB --model=LSTM --noise_ratio=$nr --v=$v --task_id=ATT
  done
done