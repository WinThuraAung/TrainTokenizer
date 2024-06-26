import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from huggingface_hub import HfApi
import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TokenizerFolder.TokenizerFile import Tokenizer
from traintokenizercode import Train_Tokenizer


dataset_url = "https://huggingface.co/spaces/5w4n/burmese-tokenizers/blob/main/dataset.csv"
# df = pd.read_csv(dataset_url)
df = dd.read_csv(dataset_url)

tokenizer = Train_Tokenizer(300)
tokenizer.concatenateString()
tokenizer.train()

def tokenize_text(text):
    return tokenizer.encode(text)

df['tokenized_text'] = df['text'].map(tokenize_text, meta=('text', 'object'))

# Save the new DataFrame to a CSV file
output_path = "processed_dataset.csv"
with ProgressBar():
    df.to_csv(output_path, index=False, single_file=True)

# Push to Huggingface
api = HfApi()
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="processed_dataset.csv",
    repo_id="RonaldAung/processed_dataset",
    repo_type="dataset"
)
