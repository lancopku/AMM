from model.base import BasicChatbotModel
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import math
from model.batch import Batch
from typing import List
from process.vocab_processor import VocabularyProcessor, GO_TOKEN, EOS_TOKEN, PAD_TOKEN
import string
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json


class Projection:
    def __init__(self, shape: List[int], scope=None, dtype=None):
        self.scope = scope
        assert len(shape) == 2
        with tf.variable_scope(scope, dtype=dtype):
            self.W = tf.get_variable('weight', shape)
            self.b = tf.get_variable('bias', shape[1:], initializer=tf.constant_initializer())

    def variables(self):
        return self.W, self.b

    def __call__(self, x):
        with tf.name_scope(self.scope):
            return x @ self.W + self.b


class Seq2SeqModel(BasicChatbotModel):
    def __init__(self, name: str = 'seq'):
        super(Seq2SeqModel, self).__init__()
        self.q_enc = None
        self.a_dec = None
        self.a_target = None
        self.a_weights = None

        self.seq_dec_outputs = None

        self.output_projection: Projection = None

        self.outputs = None

        self.optimizer = None

        self.seq_loss_op = None
        self.seq_train_op = None

    def _build_graph(self, mode: str):
        print('Building graph ...')
        self._build_placeholders()
        self._build_seq2seq(mode)
        if mode == 'train':
            self._build_optimize_ops()
        else:
            self._build_outputs_op()

    def _build_placeholders(self):
        with tf.name_scope('placeholder'):
            with tf.name_scope('q'):
                self.q_enc = [tf.placeholder(tf.int32, [None, ], name='encoders') for _ in
                              range(self.enc_length)]
            with tf.name_scope('a'):
                self.a_dec = [tf.placeholder(tf.int32, [None, ], name='decoders') for _ in
                              range(self.dec_length)]
                self.a_target = [tf.placeholder(tf.int32, [None, ], name='target') for _ in
                                 range(self.dec_length)]
                self.a_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                  range(self.dec_length)]

    def _build_seq2seq(self, mode: str):
        with tf.name_scope('seq2seq'):
            if 0 < self.hyperparameter('n_sampled') < self._vocab_size():
                vocab_size = self._vocab_size()
                hidden_size = self.hyperparameter('hidden_size')
                self.output_projection = Projection(
                    [hidden_size, vocab_size], scope='output_projection', dtype=tf.float32)

            n_layers = self.hyperparameter('n_layers')
            cell = tf.contrib.rnn.MultiRNNCell(
                [self._create_rnn_cell(mode) for _ in range(n_layers)])

            embedding_size = self.hyperparameter('embedding_size')
            output_projection = self.output_projection.variables() if \
                self.output_projection else None

            self.seq_dec_outputs, _ = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                self.q_enc,
                self.a_dec,
                cell,
                self._vocab_size(),
                self._vocab_size(),
                embedding_size=embedding_size,
                output_projection=output_projection,
                feed_previous=mode == 'test')

    def _create_rnn_cell(self, mode: str):
        hidden_size = self.hyperparameter('hidden_size')
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)

        if mode == 'train':
            dropout_keep_prob = self.hyperparameter('dropout')
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                input_keep_prob=1.0,
                output_keep_prob=dropout_keep_prob)

        return cell

    def _build_optimize_ops(self):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.training_parameter('learning_rate'))

        self.seq_loss_op = tf.contrib.legacy_seq2seq.sequence_loss(
            self.seq_dec_outputs,
            self.a_target,
            self.a_weights,
            self._vocab_size(),
            softmax_loss_function=self._sampled_softmax_fn if self.output_projection else None
        )
        self.seq_train_op = self.optimizer.minimize(self.seq_loss_op)

    def _build_outputs_op(self):
        if not self.output_projection:
            self.outputs = self.seq_dec_outputs
        else:
            self.outputs = [self.output_projection(o) for o in self.seq_dec_outputs]

        self.outputs = [tf.argmax(o, axis=1) for o in self.outputs]

    def _sampled_softmax_fn(self, labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        W_t = tf.transpose(self.output_projection.W)
        logits = tf.cast(logits, tf.float32)

        return tf.nn.sampled_softmax_loss(
            W_t,
            self.output_projection.b,
            labels,
            logits,
            self.hyperparameter('n_sampled'),
            self._vocab_size())

    def _train_step(self, batch: Batch):
        feed_dict = {}
        for i in range(self.enc_length):
            feed_dict[self.q_enc[i]] = batch.q_enc_seq[i]
        for i in range(self.dec_length):
            feed_dict[self.a_dec[i]] = batch.a_dec_seq[i]
            feed_dict[self.a_target[i]] = batch.a_target_seq[i]
            feed_dict[self.a_weights[i]] = batch.a_weights[i]

        _, loss = self.sess.run([self.seq_train_op, self.seq_loss_op], feed_dict=feed_dict)

        # Print training status
        if self.global_step % self.training_parameter('print_interval') == 0:
            perplexity = math.exp(float(loss) if loss < 300 else float('inf'))
            tqdm.write('----- Step %d -- Loss %.2f -- Perplexity %.2f' % (
                self.global_step, loss, perplexity))

    def evaluate(self):
        self._restore_model_settings()
        self._build_graph('test')
        self._create_session()
        self._restore_checkpoint()

        print('Start testing ...')
        batches = self._get_batches('test')
        test_samples = [qa for qa, _ in self.dataset['test_samples']]
        test_samples_text = [text for _, text in self.dataset['test_samples']]
        # batches = [self._create_batch(test_samples)]
        all_inputs = []
        all_outputs = []
        all_references = []

        for batch in tqdm(batches, desc='Testing'):
            feed_dict = {}
            for i in range(self.enc_length):
                feed_dict[self.q_enc[i]] = batch.q_enc_seq[i]
            feed_dict[self.a_dec[0]] = batch.a_dec_seq[0]

            [outputs] = self.sess.run([self.outputs], feed_dict=feed_dict)
            all_outputs += np.transpose(np.array(outputs)).tolist()
            all_inputs += np.transpose(np.array(batch.q_enc_seq)).tolist()
            all_references += np.transpose(np.array(batch.a_target_seq)).tolist()

        # self._write_test_samples_literal(test_samples_text, all_outputs)
        # self._write_test_samples_results(all_outputs)
        # self._wirte_test_literal(all_inputs, all_outputs)
        self._write_evaluation_results(all_outputs, all_references)

    def _write_test_samples_results(self, outputs: List[List[int]]):
        results = {}
        outputs = [self._ids2tokens(tokens) for tokens in outputs]

        grams = {1: set(), 2: set(), 3: set()}
        for g in grams:
            for tokens in outputs:
                for i in range(len(tokens)):
                    if i + g >= len(tokens):
                        break
                    grams[g].add(tuple(tokens[i:i + g]))
        results.update(dict(('{}-gram'.format(n), len(v)) for n, v in grams.items()))

        path = os.path.join(self._config['model']['save_path'], self.name)
        result_path = os.path.join(path, 'test_samples_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)

    def _write_evaluation_results(self, outputs: List[List[int]], references: List[List[int]] = None):
        results = {}
        outputs = [self._ids2tokens(tokens) for tokens in outputs]
        references = [self._ids2tokens(tokens) for tokens in references]

        weights = [
            (1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (0.33, 0.33, 0.33, 0),
            (0.25, 0.25, 0.25, 0.25)
        ]

        smoothing_fn = SmoothingFunction().method1
        pairs = tqdm([i for i in zip(outputs, references)], desc='Computing BLEU score')
        scores = [np.average(
            [sentence_bleu([ref], pred, weights=w, smoothing_function=smoothing_fn) for pred, ref in pairs]) for w in weights]
        for i, score in enumerate(scores):
            results['BLEU-{}'.format(i + 1)] = score

        grams = {1: set(), 2: set(), 3: set()}
        for g in grams:
            for tokens in outputs:
                for i in range(len(tokens)):
                    if i + g >= len(tokens):
                        break
                    grams[g].add(tuple(tokens[i:i + g]))
        results.update(dict(('{}-gram'.format(n), len(v)) for n, v in grams.items()))

        path = os.path.join(self._config['model']['save_path'], self.name)
        result_path = os.path.join(path, 'evaluation_results.json')
        with open(result_path, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)

    def _wirte_test_literal(self, inputs: List[List[int]], outputs: List[List[int]]):
        inputs = [self._ids2tokens(tokens, reverse=True) for tokens in inputs]
        outputs = [self._ids2tokens(tokens) for tokens in outputs]
        inputs = [self._tokens2literal(sent) for sent in inputs]
        outputs = [self._tokens2literal(sent) for sent in outputs]

        path = os.path.join(self._config['model']['save_path'], self.name)
        test_literal_path = os.path.join(path, 'test_samples.txt')
        with open(test_literal_path, 'w') as f:
            for q, a in zip(inputs, outputs):
                f.write(q + ' +++$+++ ' + a + '\n')

    def _write_test_samples_literal(self, text: List[str], outputs: List[List[int]]):
        outputs = [self._ids2tokens(tokens) for tokens in outputs]
        outputs = [self._tokens2literal(sent) for sent in outputs]

        path = os.path.join(self._config['model']['save_path'], self.name)
        test_literal_path = os.path.join(path, 'example_questions.txt')
        with open(test_literal_path, 'w') as f:
            for q, a in zip(text, outputs):
                f.write(q + ' +++$+++ ' + a + '\n')

    def _ids2tokens(self, seq: List[int], reverse=False):
        if not seq:
            return ''

        if reverse:
            seq = reversed(seq)

        vocab_processor: VocabularyProcessor = self.dataset['vocab_processor']
        sent = []
        for word_id in seq:
            word = vocab_processor.id2word(word_id)
            if word == EOS_TOKEN:
                break
            elif word not in [GO_TOKEN, PAD_TOKEN]:
                sent.append(word)

        return sent

    @staticmethod
    def _tokens2literal(sent: List[str]):
        text = ''.join(
            [' ' + t if not t.startswith("'") and t not in string.punctuation else t for t in sent])
        return text.strip().capitalize()
