from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import \
    embedding_rnn_decoder as tf_embedding_rnn_decoder
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import \
    embedding_attention_decoder as tf_embedding_attention_decoder
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.python.ops import array_ops
import copy


def embedding_rnn_encoder(encoder_inputs,
                          cell,
                          num_symbols,
                          embedding_size,
                          scope=None,
                          dtype=None):
    with variable_scope.variable_scope(scope or "embedding_rnn_encoder", dtype=dtype) as scope:
        dtype = scope.dtype

        # Note that we use a deep copy of the original cell
        encoder_cell = copy.deepcopy(cell)
        encoder_cell = core_rnn_cell.EmbeddingWrapper(
            encoder_cell,
            embedding_classes=num_symbols,
            embedding_size=embedding_size)
        encoder_outputs, encoder_state = rnn.static_rnn(
            encoder_cell, encoder_inputs, dtype=dtype)

        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = array_ops.concat(top_states, 1)

        return encoder_state, attention_states


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          scope=None):
    with variable_scope.variable_scope(scope or "embedding_rnn_decoder"):
        # Node that we use the original cell
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)

        return tf_embedding_rnn_decoder(
            decoder_inputs,
            initial_state,
            cell,
            num_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous)


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                scope=None,
                                initial_state_attention=False):
    with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
        if output_projection is None:
            cell = core_rnn_cell.OutputProjectionWrapper(cell, num_symbols)
            output_size = num_symbols

        return tf_embedding_attention_decoder(
            decoder_inputs,
            initial_state,
            attention_states,
            cell,
            num_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous,
            initial_state_attention=initial_state_attention)


def create_projection_params(size, dtype=None, scope=None):
    with variable_scope.variable_scope(scope or "projection_params") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        p_w = tf.get_variable('weights', (size, size), dtype=dtype)
        p_b = tf.get_variable('bias', (size,), dtype=dtype)

        return p_w, p_b


def states_projection(states,
                      hidden_size,
                      activate_fn=None,
                      dtype=None,
                      scope=None):
    with variable_scope.variable_scope(scope or "states_projection") as scope:
        if dtype is not None:
            scope.set_dtype(dtype)
        else:
            dtype = scope.dtype

        def state_projection(state, _scope=None):
            with variable_scope.variable_scope(_scope):
                if isinstance(state, LSTMStateTuple):
                    c_w, c_b = create_projection_params(hidden_size, dtype=dtype, scope='c')
                    c = tf.nn.xw_plus_b(state.c, c_w, c_b, name='c')
                    h_w, h_b = create_projection_params(hidden_size, dtype=dtype, scope='h')
                    h = tf.nn.xw_plus_b(state.h, h_w, h_b, name='h')
                    if activate_fn:
                        c = activate_fn(c)
                        h = activate_fn(h)
                    return LSTMStateTuple(c, h)
                else:
                    p_w, p_b = create_projection_params(hidden_size, dtype=dtype)
                    p = tf.nn.xw_plus_b(state, p_w, p_b)
                    if activate_fn:
                        p = activate_fn(p)
                    return p

        if type(states) == tuple:
            return tuple(
                state_projection(state, 'layer_{}'.format(i)) for i, state in enumerate(states))
        else:
            return state_projection(states)
