# 
# @author: Allan
#

from typing import List

class Sentence:

    def __init__(self, words: List[str], pos_tags:List[str] = None):
        self.words = words
        self.pos_tags = pos_tags

    def __len__(self):
        return len(self.words)








# if __name__ == "__main__":
#
#     words = ["a" ,"sdfsdf"]
#     sent = Sentence(words)
#
#     print(len(sent))