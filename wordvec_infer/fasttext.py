def subword_tokenizer(term, min_n=3, max_n=6):
    """
    :param term: str
        String to be tokenized
    :param min_n: int
        Minimum length of subword. Default is min_n = 3
    :param max_n: int
        Minimum length of subword. Default is max_n = 6

    It returns
    subwords: list of str
        It contains subword of "<term>".
        '<' and '>' are special symbol that represent begin and end of term.
    term_: str
        Term that attached '<' and '>'
    """

    term_ = '<%s>' % term
    len_ = len(term_)
    subwords = [term_[b:b+n] for n in range(min_n, max_n + 1)
                for b in range(len_ - n + 1)]
    return subwords, term_