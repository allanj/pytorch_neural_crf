from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	labels: List[str] = None
	prediction: List[str]  = None

