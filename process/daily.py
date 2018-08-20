import os

_TEXT_FILE_NAME = 'dialogues_text.txt'


def daily_conversations(source_dir: str):
    path = os.path.join(source_dir, _TEXT_FILE_NAME)
    with open(path, 'r') as f:
        conversations = [line.split(' __eou__')[:-1] for line in f]
        return conversations
