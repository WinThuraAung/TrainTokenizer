import sys
from datasets import load_dataset

sys.path.append('/Users/macbookpro/Desktop/BaseInternship')

from TokenizerFolder.basictokenizerfile import basic_tokenizer

class Train_Tokenizer:
    def __init__(self, vocab_size):
        self.dataset = load_dataset("json", data_files="/Users/macbookpro/Desktop/BaseInternship/oscar-2019-my-fix.json", split="train")
        self.trainer = basic_tokenizer()
        self.vocab_size = vocab_size
        self.text = ""

    def concatenateString(self):
        count = 1
        for doc in self.dataset:
            print(count)
            count += 1
            text = doc['text']
            self.text += text + "\n"

    def getText(self):
        return self.text

    def train(self):
        self.trainer.train(self.text, 300)
    
    def num_of_tokens(self, word):
        return len(self.encode(word))
    
    def encode(self, word):
        return self.trainer.encode(word)

TrainTokenizer = Train_Tokenizer(300)
TrainTokenizer.concatenateString()
test_phrase = "မင်္ဂလာပါ"
print(f"Encoded phrase: {TrainTokenizer.encode(test_phrase)}")
print(f"Number of tokens: {TrainTokenizer.num_of_tokens(test_phrase)}")