from mlbootstrap import Bootstrap
from fetch.fetch import Fetcher
from process.process import DataProcessor
from model.seq import Seq2SeqModel
from model.seq_attn import Seq2SeqAttentionModel
from model.auto_encoder import AutoEncoderModel

models = {
    'seq': Seq2SeqModel(),
    'seq-attn': Seq2SeqAttentionModel(),
    'auto': AutoEncoderModel('auto'),
    'auto-attn': AutoEncoderModel('auto-attn')
}

bootstrap = Bootstrap(
    'config.yaml',
    fetcher=Fetcher(),
    preprocessor=DataProcessor(),
    models=models
)

bootstrap.train()
