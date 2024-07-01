import sentencepiece as spm
from transformers import PreTrainedTokenizerFast

model_file = "trained_byte.model"

try:
    sp = spm.SentencePieceProcessor(model_file=model_file)
    print("Model loaded successfully with SentencePiece.")
    # Display a sample of tokens
    example_string = "မင်္ဂလာပါ"
    tokens = sp.encode_as_pieces(example_string)
    print(f"Tokens: {tokens}")
    print(f"Length: {len(tokens)}")
except Exception as e:
    print(f"Error loading model with SentencePiece: {e}")

print(sp.encode_as_pieces("ကျေးဇူးပြုပြီး မီး ပိတ် ပေးပါ"))


# tokenizer = PreTrainedTokenizerFast(tokenizer_file="/Users/macbookpro/Desktop/BaseInternship/trained_byte.model")

# Save the tokenizer
# tokenizer.save_pretrained("trained_tokenizer")

# Load the tokenizer
# loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained("trained_tokenizer")

# # Check tokenizer's performance
# example_string = "မင်္ဂလာပါ"
# tokens = loaded_tokenizer.tokenize(example_string)

# print(f"Example string: {example_string}")
# print(f"Number of tokens: {len(tokens)}")
# print(f"Tokens: {tokens}")
