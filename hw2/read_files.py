import glob
import tensorflow as tf
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util
import os

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
    summary_dir = "../../data/q1_bc_ant_Ant-v4_15-04-2023_22-18-15"

    """ data_lb = read_q1_data('lb')
    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'CartPole-v0' in split and batch in split:
            config_list = split[split.index(batch):split.index('CartPole-v0')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0] """




    for filename in os.listdir(summary_dir):
        path = os.path.join(summary_dir, filename)

    X, Y = get_section_results(path)
    for i, (x, y) in enumerate(zip(X, Y)):
        print('Iteration {:d} | Train steps: {:d} | Return: {}'.format(i, int(x), y))