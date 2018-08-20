from typing import List, Tuple
import nltk
from mlbootstrap.preprocess import BasicPreprocessor
import os
from process.cornell import cornell_conversations
from process.daily import daily_conversations
from process.vocab_processor import VocabularyProcessor
import pickle
from pathlib import Path
from tqdm import tqdm

_DATA_SPLIT = 0.8


def _conversation_to_qa_pairs(conversation: List[str]) -> List[Tuple[str, str]]:
    qa_pairs = []
    length = len(conversation)
    for i in range(length - 1):
        q = conversation[i]
        a = conversation[i + 1]
        qa_pairs.append((q, a))
    return qa_pairs


def _tokenize(text: str) -> List[List[str]]:
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for sent in sentences:
        for i in range(len(sent)):
            sent[i] = sent[i].lower()
    return sentences


def _is_valid_sample(sample: Tuple[List[str], List[str]],
                     vocab_processor: VocabularyProcessor) -> bool:
    q, a = sample
    conditions = [
        q,
        a,
        any(not vocab_processor.is_unknown(w) for w in q),
        all(not vocab_processor.is_unknown(w) for w in a)]
    return all(conditions)


def _vecterize(words: List[str], vocab_processor: VocabularyProcessor) -> List[int]:
    return [vocab_processor.word2id(w) for w in words]


class DataProcessor(BasicPreprocessor):
    def finished(self):
        dst = self._get_dataset_node().dst
        conditions = [
            super(DataProcessor, self).finished(),
            Path(os.path.join(dst, self.__vocab_filename())).exists(),
            Path(os.path.join(dst, self.__dataset_filename())).exists()]
        return all(conditions)

    def check(self):
        super(DataProcessor, self).check()
        dst = self._get_dataset_node().dst
        if not Path(os.path.join(dst, self.__vocab_filename())).exists():
            raise FileNotFoundError(
                "Vocabulary file '{}' does not exist".format(self.__vocab_filename()))
        if not Path(os.path.join(dst, self.__dataset_filename())).exists():
            raise FileNotFoundError(
                "Dataset file '{}' does not exist".format(self.__dataset_filename()))

    def _on_next(self, src: str, dst: str, task: str):
        os.makedirs(dst, exist_ok=True)

        full_set_fp = os.path.join(dst, 'full.dataset')
        if not Path(full_set_fp).exists():
            fn = {
                'cornell': cornell_conversations,
                'daily': daily_conversations
            }
            conversations = fn[task](src)
            qa_pairs = [qa for c in conversations for qa in _conversation_to_qa_pairs(c)]
            qa_pairs = tqdm(qa_pairs, desc='Tokenizing QA pairs', leave=False)
            tokenized_qa_pairs = [(_tokenize(q), _tokenize(a)) for q, a in qa_pairs if q and a]

            with open(full_set_fp, 'wb') as f:
                pickle.dump(tokenized_qa_pairs, f, -1)
            print('Saved full dataset.')
        else:
            with open(full_set_fp, 'rb') as f:
                tokenized_qa_pairs = pickle.load(f)

        train = self.__process_train_set(tokenized_qa_pairs, dst)
        test = self.__process_test_set(tokenized_qa_pairs, dst)

        data = {'train': train, 'test': test}
        dataset_filename = self.__dataset_filename()

        with open(os.path.join(dst, dataset_filename), 'wb') as f:
            pickle.dump(data, f, -1)

    def __process_train_set(self, tokenized_qa_pairs, dst: str):
        n_samples = len(tokenized_qa_pairs)
        train = tokenized_qa_pairs[:round(n_samples * _DATA_SPLIT)]
        train = tqdm(train, desc='Flattening QA pairs in training set', leave=False)
        train = [
            (self.__flatten_tokenized_utterance(q, reverse=True),
             self.__flatten_tokenized_utterance(a, reverse=False)) for q, a in train]

        min_frequency = self.hyperparameter('min_frequency')
        vocab_processor = VocabularyProcessor(min_frequency=min_frequency)
        for q, a in tqdm(train, desc='Fitting vocabulary', leave=False):
            for word in q + a:
                vocab_processor.add(word)
        vocab_processor.fit()
        vocab_filename = self.__vocab_filename()
        vocab_processor.save(os.path.join(dst, vocab_filename))

        train = tqdm(train, desc='Vectorizing training samples', leave=False)
        train = [(_vecterize(q, vocab_processor), _vecterize(a, vocab_processor)) for q, a in train
                 if _is_valid_sample((q, a), vocab_processor)]

        return train

    def __process_test_set(self, tokenized_qa_pairs, dst: str):
        n_samples = len(tokenized_qa_pairs)
        test = tokenized_qa_pairs[round(n_samples * _DATA_SPLIT):]
        test = tqdm(test, desc='Flattening QA pairs in testing set', leave=False)
        test = [
            (self.__flatten_tokenized_utterance(q, reverse=True),
             self.__flatten_tokenized_utterance(a, reverse=False)) for q, a in test]

        vocab_processor = VocabularyProcessor()
        vocab_filename = self.__vocab_filename()
        vocab_processor.restore(os.path.join(dst, vocab_filename))

        test = tqdm(test, desc='Vectorizing test samples', leave=False)
        test = [(_vecterize(q, vocab_processor), _vecterize(a, vocab_processor)) for q, a in test
                if _is_valid_sample((q, a), vocab_processor)]

        return test

    def __flatten_tokenized_utterance(self, utterance: List[List[str]], reverse=False) -> List[str]:
        flat = []

        if reverse:
            utterance = reversed(utterance)

        for sent in utterance:
            max_sent_length = self.hyperparameter('max_sent_length')
            if len(flat) + len(sent) <= max_sent_length:
                if reverse:
                    flat = sent + flat
                else:
                    flat = flat + sent
            else:
                break

        return flat

    def __vocab_filename(self) -> str:
        max_sent_length = self.hyperparameter('max_sent_length')
        min_frequency = self.hyperparameter('min_frequency')
        return 'max_sent_length{}-min_frequency{}.vocab'.format(max_sent_length, min_frequency)

    def __dataset_filename(self) -> str:
        max_sent_length = self.hyperparameter('max_sent_length')
        min_frequency = self.hyperparameter('min_frequency')
        return 'max_sent_length{}-min_frequency{}.dataset'.format(max_sent_length, min_frequency)

    def _load_dataset(self):
        dst = self._get_dataset_node().dst
        with open(os.path.join(dst, self.__dataset_filename()), 'rb') as f:
            data = pickle.load(f)

        vocab_processor = VocabularyProcessor()
        vocab_processor.restore(os.path.join(dst, self.__vocab_filename()))
        data['vocab_processor'] = vocab_processor

        # some fixed testing samples in 'test_samples.txt'
        with open(os.path.join(dst, 'test_samples.txt'), 'r') as f:
            texts = [line[:-1] for line in f]
            questions = [_tokenize(text) for text in texts]
            questions = [self.__flatten_tokenized_utterance(q, reverse=True) for q in questions]
            questions = [_vecterize(q, vocab_processor) for q in questions]
        test_samples = [((q, []), text) for q, text in zip(questions, texts) if q]
        data['test_samples'] = test_samples

        print('Loaded dataset: {} words, {} training samples, {} testing samples'.format(
            vocab_processor.size(), len(data['train']), len(data['test'])))

        return data
