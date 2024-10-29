import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import matplotlib.pyplot as plt


def analyze_text_data(df):
    """Analyze the text data before tokenization"""
    print("\nText Data Analysis:")
    print(f"Total rows in dataframe: {len(df)}")

    for col in ['title', 'text']:
        null_count = df[col].isnull().sum()
        empty_count = df[col].fillna('').str.strip().eq('').sum()
        avg_length = df[col].fillna('').str.len().mean()

        print(f"\n{col.upper()} Analysis:")
        print(f"- Null values: {null_count} ({null_count / len(df) * 100:.2f}%)")
        print(f"- Empty strings: {empty_count} ({empty_count / len(df) * 100:.2f}%)")
        print(f"- Average character length: {avg_length:.2f}")
        print(f"- Sample of first few {col}s:")
        for idx, text in enumerate(df[col].fillna('').head(3)):
            print(f"  {idx + 1}. {text[:100]}...")


def analyze_tokenization(tokenizer, text, label=""):
    """Analyze tokenization for a single piece of text"""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    print(f"\n{label} Tokenization Example:")
    print(f"Original text: {text[:100]}...")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of token IDs: {len(token_ids)}")
    print(f"First 10 tokens: {tokens[:10]}")
    print(f"First 10 token IDs: {token_ids[:10]}")

    return tokens, token_ids


def analyze_batch_encoding(encoded_output, label=""):
    """Analyze batch encoding output"""
    print(f"\n{label} Batch Encoding Analysis:")
    print("Keys in encoded output:", list(encoded_output.keys()))

    input_ids = encoded_output['input_ids']
    attention_mask = encoded_output['attention_mask']
    token_type_ids = encoded_output['token_type_ids']

    print(f"\nStatistics:")
    print(f"- Number of sequences: {len(input_ids)}")
    print(f"- Max sequence length: {max(len(seq) for seq in input_ids)}")
    print(f"- Min sequence length: {min(len(seq) for seq in input_ids)}")
    print(f"- Average sequence length: {np.mean([len(seq) for seq in input_ids]):.2f}")

    print("\nFirst sequence example:")
    print(f"- Input IDs: {input_ids[0][:10]}...")
    print(f"- Attention Mask: {attention_mask[0][:10]}...")
    print(f"- Token Type IDs: {token_type_ids[0][:10]}...")

    return {
        'lengths': [len(seq) for seq in input_ids],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids
    }


def plot_length_distribution(lengths, title):
    """Plot length distribution of tokenized sequences"""
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50)
    plt.title(f'Distribution of Sequence Lengths - {title}')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.show()


if __name__ == "__main__":
    print("Loading data and tokenizer...")
    text_dataframe = pd.read_csv(
        r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe.csv",
        low_memory=False
    )
    print(f"Loaded dataframe with shape: {text_dataframe.shape}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
    print(f"Loaded tokenizer: {tokenizer.__class__.__name__}")
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
    print(f"Maximum sequence length: {tokenizer.model_max_length}")

    # Analyze raw text data
    analyze_text_data(text_dataframe)

    # Analyze tokenization for a single example
    sample_title = text_dataframe['title'].fillna('').iloc[0]
    sample_text = text_dataframe['text'].fillna('').iloc[0]

    title_tokens, title_ids = analyze_tokenization(tokenizer, sample_title, "Title")
    text_tokens, text_ids = analyze_tokenization(tokenizer, sample_text, "Text")

    print("\nProcessing batch tokenization...")
    # Batch encode titles
    print("\nEncoding titles...")
    encoded1 = tokenizer.batch_encode_plus(
        text_dataframe["title"].fillna("").tolist(),
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    title_analysis = analyze_batch_encoding(encoded1, "Title")

    # Batch encode texts
    print("\nEncoding main texts...")
    encoded2 = tokenizer.batch_encode_plus(
        text_dataframe["text"].fillna("").tolist(),
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    text_analysis = analyze_batch_encoding(encoded2, "Text")

    # Plot length distributions
    print("\nGenerating length distribution plots...")
    plot_length_distribution(title_analysis['lengths'], "Titles")
    plot_length_distribution(text_analysis['lengths'], "Main Texts")

    # Additional statistics
    print("\nOverall Statistics:")
    print(f"Total sequences processed: {len(text_dataframe)}")
    print(f"Total tokens in titles: {sum(title_analysis['lengths'])}")
    print(f"Total tokens in main texts: {sum(text_analysis['lengths'])}")
    print(f"Average tokens per title: {np.mean(title_analysis['lengths']):.2f}")
    print(f"Average tokens per text: {np.mean(text_analysis['lengths']):.2f}")

    # Check for truncated sequences
    title_truncated = sum(1 for l in title_analysis['lengths'] if l >= tokenizer.model_max_length)
    text_truncated = sum(1 for l in text_analysis['lengths'] if l >= tokenizer.model_max_length)

    print("\nTruncation Analysis:")
    print(f"Titles truncated: {title_truncated} ({title_truncated / len(text_dataframe) * 100:.2f}%)")
    print(f"Texts truncated: {text_truncated} ({text_truncated / len(text_dataframe) * 100:.2f}%)")