from model.seq import Seq2SeqModel
import tensorflow as tf


class Seq2SeqAttentionModel(Seq2SeqModel):
    def __init__(self, name: str = 'seq-attn'):
        super(Seq2SeqAttentionModel, self).__init__(name)

    def _build_seq2seq(self, mode: str):
        super(Seq2SeqAttentionModel, self)._build_seq2seq(mode)

        with tf.name_scope('seq2seq'):
            n_layers = self.hyperparameter('n_layers')
            cell = tf.contrib.rnn.MultiRNNCell(
                [self._create_rnn_cell(mode) for _ in range(n_layers)])

            embedding_size = self.hyperparameter('embedding_size')
            output_projection = self.output_projection.variables() if \
                self.output_projection else None

            self.seq_dec_outputs, _ = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.q_enc,
                self.a_dec,
                cell,
                self._vocab_size(),
                self._vocab_size(),
                embedding_size,
                output_projection=output_projection,
                feed_previous=mode == 'test')
