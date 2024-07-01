import pandas as pd
import sentencepiece as spm

csv_file = "https://huggingface.co/spaces/5w4n/burmese-tokenizers/raw/main/dataset.csv"
df = pd.read_csv(csv_file)

model_file = "trained_byte.model"
tokenizer = spm.SentencePieceProcessor(model_file=model_file)

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

df['trained_byte'] = df['text'].apply(tokenize_text)


output_csv_file = "updated_dataset.csv"
df.to_csv(output_csv_file, index=False)

print("Tokenization complete. Updated CSV saved to:", output_csv_file)
