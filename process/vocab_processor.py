import pickle

GO_TOKEN = '<go>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'

SPECIAL_TOKENS = {
    GO_TOKEN, EOS_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN
}


class VocabularyProcessor:
    def __init__(self, min_frequency: int = 0):
        self._word2id = {}
        self._id2word = {}
        self._word_frequency = {}
        self._min_frequency = min_frequency

        self.add(GO_TOKEN)
        self.add(EOS_TOKEN)
        self.add(PAD_TOKEN)
        self.add(UNKNOWN_TOKEN)

    def size(self) -> int:
        return len(self._word2id)

    def add(self, word: str, frequency: int = 1):
        word = word.lower()
        if word in self._word2id:
            self._word_frequency[word] += frequency
        else:
            word_id = len(self._word2id)
            self._word2id[word] = word_id
            self._id2word[word_id] = word
            self._word_frequency[word] = frequency

    def fit(self):
        vocab_processor = VocabularyProcessor()
        for word, frequency in self._word_frequency.items():
            if word not in SPECIAL_TOKENS and frequency >= self._min_frequency:
                vocab_processor.add(word, frequency)

        min_frequency = self._min_frequency
        self.__dict__.update(vocab_processor.__dict__)
        self._min_frequency = min_frequency

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f, -1)

    def restore(self, filename: str):
        with open(filename, 'rb') as f:
            self.__dict__.update(pickle.load(f))

    def word2id(self, word: str) -> int:
        word = word.lower()
        return self._word2id.get(word, self._word2id[UNKNOWN_TOKEN])

    def id2word(self, word_id: int) -> str:
        return self._id2word[word_id]

    def word_frequency(self, word: str) -> int:
        word = word.lower()
        return self._word_frequency[word] if word in self._word2id else 0

    def is_unknown(self, word: str) -> bool:
        word = word.lower()
        return word not in self._word2id or word == UNKNOWN_TOKEN
