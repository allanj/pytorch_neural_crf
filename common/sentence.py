# 
# @author: Allan
#

from typing import List

class Sentence:
    """
    The class for the input sentence
    """

    def __init__(self, words: List[str], pos_tags:List[str] = None):
        """

        :param words:
        :param pos_tags: By default, it is not required to have the pos tags, in case you need it/
        """
        self.words = words
        self.pos_tags = pos_tags

    def __len__(self):
        return len(self.words)


