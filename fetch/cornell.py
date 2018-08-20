import fetch.util
from pathlib import Path
import os

_CORPUS_URL = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
_ZIP_FILENAME = 'cornell_movie_dialogs_corpus.zip'
_ORIGINAL_FOLDER = 'cornell movie-dialogs corpus'


def fetch_cornell(target_path):
    if not Path(target_path).is_dir():
        zip_fp = os.path.join(fetch.util.DATA_ROOT, _ZIP_FILENAME)
        fetch.util.download(_CORPUS_URL, zip_fp)
        fetch.util.unzip(fetch.util.DATA_ROOT, _ZIP_FILENAME)
        original_dir = os.path.join(fetch.util.DATA_ROOT, _ORIGINAL_FOLDER)
        os.rename(original_dir, target_path)
    for filename in os.listdir(target_path):
        if filename not in {'movie_conversations.txt', 'movie_lines.txt', 'README.txt'}:
            os.remove(os.path.join(target_path, filename))
