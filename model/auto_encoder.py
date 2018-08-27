from model.seq import Seq2SeqModel
from model.tf_rnn_helper import *
from model.batch import Batch
from tqdm import tqdm
import math


class AutoEncoderModel(Seq2SeqModel):
    def __init__(self, name: str = 'auto'):
        super(AutoEncoderModel, self).__init__()

        self.q_dec = None
        self.q_target = None
        self.q_weights = None
        self.a_enc = None

        self.q_dec_outputs = None
        self.a_dec_outputs = None

        self.q_loss_op = None
        self.q_train_op = None
        self.a_loss_op = None
        self.a_train_op = None

    def _build_placeholders(self):
        super(AutoEncoderModel, self)._build_placeholders()

        with tf.name_scope('placeholder'):
            with tf.name_scope('q'):
                self.q_dec = [tf.placeholder(tf.int32, [None, ], name='decoders') for _ in
                              range(self.dec_length)]
                self.q_target = [tf.placeholder(tf.int32, [None, ], name='target') for _ in
                                 range(self.dec_length)]
                self.q_weights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in
                                  range(self.dec_length)]
            with tf.name_scope('a'):
                self.a_enc = [tf.placeholder(tf.int32, [None, ], name='encoders') for _ in
                              range(self.enc_length)]

    def _build_seq2seq(self, mode: str):
        super(AutoEncoderModel, self)._build_seq2seq(mode)

        with tf.name_scope('seq2seq'):
            n_layers = self.hyperparameter('n_layers')

            q_cell = tf.contrib.rnn.MultiRNNCell(
                [self._create_rnn_cell(mode) for _ in range(n_layers)])
            a_cell = tf.contrib.rnn.MultiRNNCell(
                [self._create_rnn_cell(mode) for _ in range(n_layers)])

            embedding_size = self.hyperparameter('embedding_size')
            output_projection = self.output_projection.variables() if \
                self.output_projection else None

            q_states, q_attn_states = embedding_rnn_encoder(
                self.q_enc,
                q_cell,
                self._vocab_size(),
                embedding_size,
                scope='q_encoder')
            self.q_dec_outputs, _ = embedding_rnn_decoder(
                self.q_dec,
                q_states,
                q_cell,
                self._vocab_size(),
                embedding_size,
                output_projection=output_projection,
                feed_previous=mode == 'test',
                scope='q_decoder')

            a_states, _ = embedding_rnn_encoder(
                self.a_enc,
                a_cell,
                self._vocab_size(),
                embedding_size,
                scope='a_encoder')
            self.a_dec_outputs, _ = embedding_rnn_decoder(
                self.a_dec,
                a_states,
                a_cell,
                self._vocab_size(),
                embedding_size,
                output_projection=output_projection,
                feed_previous=mode == 'test',
                scope='a_decoder')

            states_p = states_projection(
                q_states,
                self.hyperparameter('hidden_size'),
                activate_fn=tf.nn.tanh,
                dtype=tf.float32)

            if 'attn' in self.name:  # with attention
                self.seq_dec_outputs, _ = embedding_attention_decoder(
                    self.a_dec,
                    states_p,
                    q_attn_states,
                    a_cell,
                    self._vocab_size(),
                    embedding_size,
                    output_projection=output_projection,
                    feed_previous=mode == 'test',
                    scope='seq_decoder')
            else:
                self.seq_dec_outputs, _ = embedding_rnn_decoder(
                    self.a_dec,
                    states_p,
                    a_cell,
                    self._vocab_size(),
                    embedding_size,
                    output_projection=output_projection,
                    feed_previous=mode == 'test',
                    scope='seq_decoder')

    def _build_optimize_ops(self):
        super(AutoEncoderModel, self)._build_optimize_ops()

        self.q_loss_op = tf.contrib.legacy_seq2seq.sequence_loss(
            self.q_dec_outputs,
            self.q_target,
            self.q_weights,
            self._vocab_size(),
            softmax_loss_function=self._sampled_softmax_fn if self.output_projection else None)
        self.a_loss_op = tf.contrib.legacy_seq2seq.sequence_loss(
            self.a_dec_outputs,
            self.a_target,
            self.a_weights,
            self._vocab_size(),
            softmax_loss_function=self._sampled_softmax_fn if self.output_projection else None)

        self.q_train_op = self.optimizer.minimize(self.q_loss_op)
        self.a_train_op = self.optimizer.minimize(self.a_loss_op)

    def _train_step(self, batch: Batch):
        feed_dict = {}
        for i in range(self.enc_length):
            feed_dict[self.q_enc[i]] = batch.q_enc_seq[i]
            feed_dict[self.a_enc[i]] = batch.a_enc_seq[i]
        for i in range(self.dec_length):
            feed_dict[self.q_dec[i]] = batch.q_dec_seq[i]
            feed_dict[self.q_target[i]] = batch.q_target_seq[i]
            feed_dict[self.q_weights[i]] = batch.q_weights[i]
            feed_dict[self.a_dec[i]] = batch.a_dec_seq[i]
            feed_dict[self.a_target[i]] = batch.a_target_seq[i]
            feed_dict[self.a_weights[i]] = batch.a_weights[i]

        _, q_loss, _, a_loss = self.sess.run(
            [self.q_train_op, self.q_loss_op, self.a_train_op, self.a_loss_op],
            feed_dict=feed_dict)

        _, loss = self.sess.run([self.seq_train_op, self.seq_loss_op], feed_dict=feed_dict)

        # Print training status
        if self.global_step % self.training_parameter('print_interval') == 0:
            perplexity = math.exp(float(loss) if loss < 300 else float('inf'))
            tqdm.write(
                '----- Step %d -- Q Loss %.2f -- A Loss %.2f -- Loss %.2f -- Perplexity %.2f' % (
                    self.global_step, q_loss, a_loss, loss, perplexity))
