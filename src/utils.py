import os
import datetime
import numpy as np
import pickle
import random
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT_DIR = 'xxx/your_path'


def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")


def time_record():
    '''
    return time stamp
    '''
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)

    return str(year) + '-' + str(month) + str(day) + '-' + str(hour) + str(minute)


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().float() / y.size(0)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    f.close()


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def seed_torch(seed) -> None:
    """ set random seed for all related packages
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'backends'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # rqh 0612 add, to ensure further reproducibility


def get_model_save_path(model_tag: str):
    print('DEFINE YOUR llm name & path or ADD new model & path')
    model_path = ''
    
    return model_path