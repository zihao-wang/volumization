from argparse import ArgumentParser
import os
import json
import matplotlib.pyplot as plt
import pandas as pd

args = ArgumentParser()
args.add_argument('--task_id', type=str, default='ATT')
args.add_argument('--dataset', type=str, default='IMDB')
args.add_argument('--model', type=str, default='LSTM')


def is_meta(f):
    return f.split('.')[-1] == 'meta'


def is_log(f):
    return f.split('.')[-1] == 'log'


row_of_interests = ['v', 'noise_ratio']


if __name__ == "__main__":
    params = args.parse_args()
    log_dir_name = os.path.join('log', params.dataset)

    file_names = os.listdir(log_dir_name)
    print(file_names)

    from collections import defaultdict
    cases = defaultdict(dict)
    for f in file_names:
        task_id, k1, k2, tail = f.split('-')
        if task_id == params.task_id:
            if is_log(tail):
                cases[k1+k2]['log'] = pd.read_csv(os.path.join(log_dir_name, f))
            if is_meta(tail):
                with open(os.path.join(log_dir_name, f), mode='rt') as metafile:
                    cases[k1+k2]['meta'] = json.load(metafile)

    df = pd.DataFrame(columns=row_of_interests + ['test_acc'])
    for case in cases.values():
        new_dict = {roi: case['meta'][roi] for roi in row_of_interests}
        best_val_acc = 0
        test_acc_at_best_val_acc = 0
        log_df = case['log']
        for _, (val_acc, test_acc) in log_df[['val_acc', 'test_acc']].iterrows():
            print(val_acc, test_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val_acc = test_acc
        new_dict['test_acc'] = test_acc_at_best_val_acc
        print(new_dict)
        df = df.append(new_dict, ignore_index=True)
    df.to_csv(os.path.join(log_dir_name, params.task_id + '.csv'))
