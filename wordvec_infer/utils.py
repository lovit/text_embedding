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

class Word2VecCorpus:
    def __init__(self, path, num_doc=-1):
        self.path = path
        self.num_doc = num_doc

    def __iter__(self):
        with open(self.path, encoding='utf-8') as f:
            for i, doc in enumerate(f):
                if self.num_doc > 0 and i >= self.num_doc:
                    break
                doc = doc.strip()
                if not doc:
                    continue
                self._tokenize(doc)

    def _tokenize(self, doc):
        for sent in doc.split('  '):
            if sent:
                yield word2vec_tokenizer(sent)

def word2vec_tokenizer(sent):
    return sent.split()