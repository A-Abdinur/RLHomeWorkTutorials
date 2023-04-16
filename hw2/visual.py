# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b <b> -lr <r> -rtg --nn_baseline --exp_name q4_search_b<b>_lr<r>_rtg_nnbaseline
for b in [10000, 30000, 50000]:
    for lr in [0.005, 0.01, 0.02]:
        print(f'''python cs285/scripts/run_hw2.py --env_name HalfCheetah-v2 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b {b} -lr {lr} -rtg --nn_baseline --exp_name q4_search_b{b}_lr{lr}_rtg_nnbaseline''')

import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util
import os


figsize=(5.7, 3)
export_dir = os.path.join('soln_pdf', 'figures')

sns.set_theme()
sns.set_context("paper")

def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)

def get_section_results(path):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    
    for event in my_summary_iterator(path):
        for value in event.summary.value:
            if value.tag == 'Train_EnvstepsSoFar':
                X.append(value.simple_value)
            elif value.tag == 'Eval_AverageReturn':
                Y.append(value.simple_value)
    return X, Y


if __name__ == '__main__':
    import glob
    full_data = pd.DataFrame()
    summary_dir = "../../data/q1_bc_ant_Ant-v4_15-04-2023_22-18-15"
    for filename in os.listdir(summary_dir):
        path = os.path.join(summary_dir, filename)

    X, Y = get_section_results(path)

    data = pd.DataFrame({'Iteration': range(len(X)), 
                                    'Train_EnvstepsSoFar': X, 
                                    'Eval_AverageReturn': Y})
    data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
    full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
    print(full_data.head())
    # plt.figure(figsize=figsize)
    # sns.lineplot(data=full_data, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')

    # plt.savefig(os.path.join(export_dir, 'q1_lb.pdf'), bbox_inches='tight')