import os
import psutil

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def check_dirs(filepath):
    dirname = os.path.dirname(filepath)
    if dirname and dirname == '.' and not os.path.exists(dirname):
        os.makedirs(dirname)
        print('created {}'.format(dirname))