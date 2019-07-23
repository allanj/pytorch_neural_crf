# 
# @author: Allan
#
from common.sentence import  Sentence
from typing import List

class Instance:
    """
    This class is the basic Instance for a datasample
    """

    def __init__(self, input: Sentence, output: List[str] = None) -> None:
        """
        Constructor for the instance.
        :param input: sentence containing the words
        :param output: a list of labels
        """
        self.input = input
        self.output = output
        self.elmo_vec = None #used for loading the ELMo vector.
        self.word_ids = None
        self.char_ids = None
        self.output_ids = None

    def __len__(self):
        return len(self.input)
