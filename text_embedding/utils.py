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
    def __init__(self, path, num_doc=-1, verbose_point=-1, lowercase=True, sentence_separator='\n'):
        self.path = path
        self.num_doc = num_doc
        self.verbose_point = verbose_point
        self.lowercase = lowercase
        self.sentence_separator = sentence_separator
        self._len = 0

    def __iter__(self):
        vp = self.verbose_point
        with open(self.path, encoding='utf-8') as f:
            for i, doc in enumerate(f):
                if self.lowercase:
                    doc = doc.lower()
                if vp > 0 and i % vp == 0:
                    print('\riterating corpus ... %d lines' % (i+1), end='', flush=True)
                if self.num_doc > 0 and i >= self.num_doc:
                    break
                doc = doc.strip()
                if not doc:
                    continue
                streams = self._tokenize(i, doc)
                for stream in streams:
                    yield stream
            if vp > 0:
                print('\riterating corpus was done%s' % (' '*20))
            self._len = i + 1

    def _tokenize(self, i, doc):
        return [sent.split() for sent in doc.split(self.sentence_separator) if sent]

    def __len__(self):
        if self._len == 0:
            with open(self.path, encoding='utf-8') as f:
                for i, _ in enumerate(f):
                    continue
                self._len = (i+1)
        return self._len

class Doc2VecCorpus(Word2VecCorpus):
    def __init__(self, path, num_doc=-1, verbose_point=-1, lowercase=True, yield_label=True, sentence_separator='\n'):
        super().__init__(path, num_doc, verbose_point, lowercase, sentence_separator)
        self.yield_label = yield_label

    def _tokenize(self, i, doc):
        column = doc.split('\t')
        if len(column) == 1:
            labels = ['__doc{}__'.format(i)]
            sents = column[0].split(self.sentence_separator)
        else:
            labels = column[0].split()
            sents = []
            for col in column[1:]:
                sents += col.split(self.sentence_separator)
        if self.yield_label:
            return [(labels, tuple(sent.split())) for sent in sents if sent]
        else:
            return [tuple(sent.split()) for sent in sents if sent]

class WordVectorInferenceDecorator:
    def __init__(self, corpus, test_terms, training=True):
        self.corpus = corpus
        self.test_terms = test_terms
        self.training = training

    def __iter__(self):
        for sent in self.corpus:
            if self.training != self._sent_has_term(sent):
                yield sent

    def _sent_has_term(self, sent):
        for term in sent:
            if term in self.test_terms:
                return True
        return False