from allennlp.data import Vocabulary

"""
Stub classes before wrap allennlp's entities and required minimum of methods
for fairseq's transformer model to work

"""


class Args:
    """Stub class for arguments"""
    def __init__(self):
        pass

class Dictionary:
    """Stub class for dictionary (vocab)"""
    def __init__(self, vocab: Vocabulary, namespace: str,
                 eos: str = '@end@',
                 pad: str = '@@PADDING@@',
                 unk: str = '@@UNKNOWN@@'):
        self.unk_word, self.pad_word, self.eos_word = unk, pad, eos

        self.pad_index = vocab.get_token_index(pad, namespace)
        self.eos_index = vocab.get_token_index(eos, namespace)
        self.unk_index = vocab.get_token_index(unk, namespace)

        self._vocab_size = vocab.get_vocab_size(namespace)

        self._namespace = namespace

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def eos(self):
        """Helper to get index of end-of-sentence symbol"""
        return self.eos_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    def __len__(self):
        return self._vocab_size

    def __eq__(self, other):
        return self._namespace == other.get_namespace()

    def get_namespace(self):
        return self._namespace

