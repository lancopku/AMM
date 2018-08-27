from mlbootstrap.model import BasicModel
from model.batch import Batch
from typing import List, Tuple
from process.vocab_processor import VocabularyProcessor, GO_TOKEN, EOS_TOKEN, PAD_TOKEN
import random
import tensorflow as tf
import datetime
from tqdm import tqdm
import yaml
import os

_MODEL_STATUS_FILENAME = 'model_status.yaml'


class BasicChatbotModel(BasicModel):
    def __init__(self, name: str = 'basic_chatbot_model'):
        super(BasicChatbotModel, self).__init__()

        self.enc_length = None
        self.dec_length = None

        self.sess: tf.Session = None
        self.saver: tf.train.Saver = None

        self.global_step = 0

    def train(self):
        self._restore_model_settings()
        self._build_graph('train')
        self._create_session()
        self._restore_checkpoint()

        print('Start training ...')
        epoch = self.training_parameter('epoch')

        for e in range(1, epoch + 1):
            print()
            learning_rate = self.training_parameter('learning_rate')
            print(
                '----- Epoch {}/{} ; (learning_rate={}) -----'.format(e, epoch, learning_rate))

            tic = datetime.datetime.now()
            batches = self._get_batches('train')
            for batch in tqdm(batches, desc='Training'):
                self.global_step += 1
                self._train_step(batch)

                if self.global_step % self.training_parameter('save_interval') == 0:
                    self._save_checkpoint()

            toc = datetime.datetime.now()
            print('Epoch finished in {}'.format(toc - tic))

    def _restore_model_settings(self):
        path = os.path.join(self._config['model']['save_path'], self.name)
        os.makedirs(path, exist_ok=True)
        status_path = os.path.join(path, _MODEL_STATUS_FILENAME)

        if os.path.exists(status_path):
            with open(status_path, 'r') as stream:
                model_status = yaml.load(stream)
                self.global_step = model_status['global_step']
                self._config['hyperparameter'] = model_status['hyperparameter']

        max_sent_length = self.hyperparameter('max_sent_length')
        self.enc_length = max_sent_length
        self.dec_length = max_sent_length + 2

    def _build_graph(self, mode: str):
        raise NotImplementedError

    def _create_session(self):
        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True)
        ))
        self.saver = tf.train.Saver()

    def _restore_checkpoint(self):
        path = os.path.join(self._config['model']['save_path'], self.name)
        os.makedirs(path, exist_ok=True)
        status_path = os.path.join(path, _MODEL_STATUS_FILENAME)

        if os.path.exists(status_path):
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint found')
                exit(1)
        else:
            self.sess.run(tf.global_variables_initializer())

    def _save_checkpoint(self):
        tqdm.write("Checkpoint reached: saving model (don't stop the run) ...")

        path = os.path.join(self._config['model']['save_path'], self.name)
        os.makedirs(path, exist_ok=True)
        self._save_model_status(path)

        model_path = os.path.join(path, 'model')
        self.saver.save(self.sess, model_path, global_step=self.global_step)
        tqdm.write('Model saved.')

    def _save_model_status(self, path: str):
        model_status = {
            'hyperparameter': self._config['hyperparameter'],
            'global_step': self.global_step}
        status_path = os.path.join(path, _MODEL_STATUS_FILENAME)
        with open(status_path, 'w') as stream:
            yaml.dump(model_status, stream)

    def _train_step(self, batch: Batch):
        raise NotImplementedError

    def _vocab_size(self):
        return self.dataset['vocab_processor'].size()

    def _get_batches(self, mode: str) -> List[Batch]:
        if mode == 'train':
            random.shuffle(self.dataset[mode])
        samples = self.dataset[mode]
        batch_size = self.training_parameter('batch_size')
        samples_list = [samples[i:min(i + batch_size, len(samples))] for i in
                        range(0, len(samples), batch_size)]
        return [self._create_batch(samples) for samples in samples_list]

    def _create_batch(self, samples: List[Tuple[List, List]]) -> Batch:
        batch = Batch()
        vocab_processor: VocabularyProcessor = self.dataset['vocab_processor']
        go_id = vocab_processor.word2id(GO_TOKEN)
        eos_id = vocab_processor.word2id(EOS_TOKEN)
        pad_id = vocab_processor.word2id(PAD_TOKEN)

        for q, a in samples:
            def __pad(arr, pad_token, length, from_left=False):
                if len(arr) >= length:
                    return arr
                padding = [pad_token] * (length - len(arr))
                if from_left:
                    arr = padding + arr
                else:
                    arr = arr + padding
                return arr

            q_enc_seq = __pad(list(reversed(q)), pad_id, self.enc_length, from_left=True)
            batch.q_enc_seq.append(q_enc_seq)

            q_dec_seq = __pad([go_id] + q + [eos_id], pad_id, self.dec_length)
            batch.q_dec_seq.append(q_dec_seq)

            q_target_seq = __pad(q + [eos_id], pad_id, self.dec_length)
            batch.q_target_seq.append(q_target_seq)

            q_weights = __pad([1.0] * len(q), 0.0, self.dec_length)
            batch.q_weights.append(q_weights)

            a_enc_seq = __pad(list(reversed(a)), pad_id, self.enc_length, from_left=True)
            batch.a_enc_seq.append(a_enc_seq)

            a_dec_seq = __pad([go_id] + a + [eos_id], pad_id, self.dec_length)
            batch.a_dec_seq.append(a_dec_seq)

            a_target_seq = __pad(a + [eos_id], pad_id, self.dec_length)
            batch.a_target_seq.append(a_target_seq)

            a_weights = __pad([1.0] * len(a), 0.0, self.dec_length)
            batch.a_weights.append(a_weights)

        def __transpose(arr):
            len1 = len(arr)
            len2 = len(arr[0])
            result = []
            for _i in range(len2):
                next_arr = []
                for _j in range(len1):
                    next_arr.append(arr[_j][_i])
                result.append(next_arr)
            return result

        batch.q_enc_seq = __transpose(batch.q_enc_seq)
        batch.q_dec_seq = __transpose(batch.q_dec_seq)
        batch.q_target_seq = __transpose(batch.q_target_seq)
        batch.q_weights = __transpose(batch.q_weights)
        batch.a_enc_seq = __transpose(batch.a_enc_seq)
        batch.a_dec_seq = __transpose(batch.a_dec_seq)
        batch.a_target_seq = __transpose(batch.a_target_seq)
        batch.a_weights = __transpose(batch.a_weights)

        return batch
