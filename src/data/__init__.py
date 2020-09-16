
import collections
Instance = collections.namedtuple('Instance', 'words ori_words labels')
Instance.__new__.__defaults__ = (None,) * 3

from src.data.transformers_dataset import TransformersNERDataset
