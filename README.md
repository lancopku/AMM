# Auto-Encoder Matching Model

The code for "An Auto-Encoder Matching Model for Learning Utterance-Level Semantic Dependency in Dialogue Generation"

## Requirements

- Python 3
- Tensorflow >= 1.8
- mlbootstrap == 0.02

## Data Preparation

- Get the DailyDialog dataset at http://yanran.li/dailydialog.html
- Unzip the downloaded file
- Move `dialogues_text.txt` to `data/source/daily/dialogues_text.txt`

To use your own data, create a folder `data/source/<dataset-name>/` and place the original data in the directory.
Then write a parsing script (you can refer to [daily.py](./process/daily.py)) and update the `config.yaml` to include the new data path.

## Training

`python play.py`

## Evaluation

Change the last line in `play.py` to `bootstrap.evaluate()` and run `python play.py`

## Hyperparameters

You can change the hyperparameters in `config.yaml` according to your needs.