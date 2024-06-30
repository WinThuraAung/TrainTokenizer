import sentencepiece as spm
import os
import json
from transformers import PreTrainedTokenizerFast

input_json_path = "oscar-2019-my-fix.json"
output_text_path = "oscar-2019-my-fix.txt"


with open(input_json_path, 'r', encoding='utf-8') as json_file:
    with open(output_text_path, 'w', encoding='utf-8') as text_file:
        for line in json_file:
            data = json.loads(line)
            text_file.write(data['text'] + '\n')  

# Options for training the SentencePiece model
options = dict(
    input="oscar-2019-my-fix.txt",
    input_format="text",
    model_prefix="trained_byte",
    model_type="bpe",
    vocab_size=2000,
    normalization_rule_name="identity",
    remove_extra_whitespaces=False,
    input_sentence_size=200000000,
    max_sentence_length=4192,
    seed_sentencepiece_size=1000000,
    shuffle_input_sentence=True,
    character_coverage=0.99995,
    byte_fallback=True,
    split_digits=True,
    split_by_unicode_script=True,
    split_by_whitespace=True,
    split_by_number=True,
    max_sentencepiece_length=16,
    add_dummy_prefix=True,
    allow_whitespace_only_pieces=True,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=-1,
    num_threads=os.cpu_count(),
)

# Train the SentencePiece model
spm.SentencePieceTrainer.train(**options)
print("Done")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="trained_byte.model")

# Save the tokenizer
tokenizer.save_pretrained("trained_tokenizer")

# Load the tokenizer
loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained("trained_tokenizer")

# Check tokenizer's performance
example_string = "မင်္ဂလာပါ"
tokens = loaded_tokenizer.tokenize(example_string)

print(f"Example string: {example_string}")
print(f"Number of tokens: {len(tokens)}")
print(f"Tokens: {tokens}")

