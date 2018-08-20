from mlbootstrap.fetch import BasicFetcher
import fetch.util
from fetch.cornell import fetch_cornell
import os


class Fetcher(BasicFetcher):
    def fetch(self):
        os.makedirs(fetch.util.DATA_ROOT, exist_ok=True)
        fetch_fn = {
            'cornell': fetch_cornell
        }
        fetch_fn[task](self._get_dataset_node().src)
