import pandas as pd
from Tokenizer.Basic_Tokenizer import basic_tokenizer
from huggingface_hub import HfApi
from datasets import load_dataset
from TrainTokenizer import TrainTokenizer

dataset_url = "https://huggingface.co/spaces/5w4n/burmese-tokenizers/blob/main/dataset.csv"
df = pd.read_csv(dataset_url)

tokenizer = TrainTokenizer(300)
tokenizer.concatenateString()
tokenizer.train()


df['tokenized_text'] = df['text'].apply(tokenizer.encode)

# Save the new DataFrame to a CSV file
output_path = "processed_dataset.csv"
df.to_csv(output_path, index=False)

# Push to Huggingface
api = HfApi()
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="processed_dataset.csv",
    repo_id="RonaldAung/processed_dataset",
    repo_type="dataset"
)
