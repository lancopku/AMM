import os
import requests
import shutil

DATA_ROOT = 'data/source'


def download(url, fp):
    with requests.Session() as session:
        response = session.get(url, stream=True)
        chunk_size = 32768
        with open(fp, 'wb') as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        response.close()


def unzip(path, filename):
    fp = os.path.join(path, filename)
    shutil.unpack_archive(fp, path)
    os.remove(fp)
