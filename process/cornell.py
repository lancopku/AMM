import os
import re
import ast

_MOVIE_LINES_FILE_NAME = 'movie_lines.txt'
_MOVIE_CONVERSATIONS_FILE_NAME = 'movie_conversations.txt'
_MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
_MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]


def cornell_conversations(source_dir: str):
    id2line = {}
    path = os.path.join(source_dir, _MOVIE_LINES_FILE_NAME)
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            line = dict(zip(_MOVIE_LINES_FIELDS, line.split(' +++$+++ ')))
            text = re.sub('(\n)|(<u>)|(</u>)|(\[\d\])', '', line['text'])
            id2line[line['lineID']] = text

    conversations = []
    path = os.path.join(source_dir, _MOVIE_CONVERSATIONS_FILE_NAME)
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            conv = dict(zip(_MOVIE_CONVERSATIONS_FIELDS, line.split(' +++$+++ ')))
            line_ids = ast.literal_eval(conv['utteranceIDs'])
            conversations.append([id2line[i] for i in line_ids])

    return conversations
