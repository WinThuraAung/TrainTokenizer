import pandas as pd
from transformers import PreTrainedTokenizerFast

csv_file = "https://huggingface.co/spaces/5w4n/burmese-tokenizers/raw/main/dataset.csv"
df = pd.read_csv(csv_file)

tokenizer = "trained_byte.model"
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer)

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

df['tokenized_text'] = df['text'].apply(tokenize_text)

output_csv_file = "updated_dataset.csv"
df.to_csv(output_csv_file, index=False)

print("Tokenization complete. Updated CSV saved to:", output_csv_file)