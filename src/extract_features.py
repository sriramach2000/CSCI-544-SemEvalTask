import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize

# Download required NLTK data
print("Checking NLTK resources...")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')


def contain_text(text_df, pair_id: str):
    """Check if both texts in pair exist in dataframe."""
    try:
        text_id1, text_id2 = pair_id.split("_")
        result = (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)
        return result
    except Exception as e:
        print(f"Error checking pair {pair_id}: {str(e)}")
        return False


def analyze_vectors(t1_vectors, t2_vectors, name=""):
    """Analyze vector properties for debugging."""
    print(f"\nVector Analysis ({name}):")
    print(f"- Shape t1: {t1_vectors.shape}, t2: {t2_vectors.shape}")
    print(f"- Non-zero elements t1: {t1_vectors.nnz}, t2: {t2_vectors.nnz}")
    print(f"- Density t1: {t1_vectors.nnz / (t1_vectors.shape[0] * t1_vectors.shape[1]):.4f}")
    print(f"- Density t2: {t2_vectors.nnz / (t2_vectors.shape[0] * t2_vectors.shape[1]):.4f}")


def jaccard(t1_vectors, t2_vectors):
    """Calculate Jaccard similarity."""
    num = t1_vectors.minimum(t2_vectors).sum(axis=1)
    den = t1_vectors.maximum(t2_vectors).sum(axis=1)
    similarity = num / (den + 1e-10)
    return similarity


def dice(t1_vectors, t2_vectors):
    """Calculate Dice similarity."""
    num = 2 * t1_vectors.minimum(t2_vectors).sum(axis=1)
    den = t1_vectors.sum(axis=1) + t2_vectors.sum(axis=1)
    similarity = num / (den + 1e-10)
    return similarity


def cosine(t1_vectors, t2_vectors):
    """Calculate cosine similarity."""
    t1_vectors = normalize(t1_vectors)
    t2_vectors = normalize(t2_vectors)
    return t1_vectors.multiply(t2_vectors).sum(axis=1)


def create_match_features(df, target_cols: str):
    """Create similarity features between text pairs."""
    print("\nCreating matching features...")
    res = df.copy()

    for target_col in target_cols:
        print(f"\nProcessing column: {target_col}")
        print(f"Sample texts from {target_col}_1:", df[f"{target_col}_1"].iloc[:2].tolist())
        print(f"Sample texts from {target_col}_2:", df[f"{target_col}_2"].iloc[:2].tolist())

        try:
            NLTK_STOPWORDS = set(nltk.corpus.stopwords.words("english"))
            print(f"Number of stopwords: {len(NLTK_STOPWORDS)}")
        except LookupError:
            print("Warning: Could not load stopwords, proceeding without them...")
            NLTK_STOPWORDS = set()

        # Configure vectorizer
        ngram_range = (1, 2)
        count_vectorizer = CountVectorizer(
            tokenizer=lambda x: x.split(),
            stop_words=NLTK_STOPWORDS,
            ngram_range=ngram_range,
            min_df=5,
            binary=True,
        )

        # Create document list
        documents = df[f"{target_col}_1"].tolist() + df[f"{target_col}_2"].tolist()
        print(f"Total documents to vectorize: {len(documents)}")

        # Fit and transform
        print("Vectorizing documents...")
        title_counts = count_vectorizer.fit_transform(documents)
        print(f"Vocabulary size: {len(count_vectorizer.vocabulary_)}")
        print(f"Count matrix shape: {title_counts.shape}")

        print("Calculating TF-IDF...")
        tfidf_transformer = TfidfTransformer()
        title_tfidfs = tfidf_transformer.fit_transform(title_counts)
        print(f"TF-IDF matrix shape: {title_tfidfs.shape}")

        # Split vectors
        t1_counts, t2_counts = title_counts[: len(df)], title_counts[len(df):]
        t1_tfidfs, t2_tfidfs = title_tfidfs[: len(df)], title_tfidfs[len(df):]

        analyze_vectors(t1_counts, t2_counts, f"{target_col} counts")
        analyze_vectors(t1_tfidfs, t2_tfidfs, f"{target_col} tf-idf")

        print(f"\nCalculating similarities for {target_col}...")
        suffix = target_col

        # Calculate similarities
        res[f"jaccard_count_{suffix}"] = jaccard(t1_counts, t2_counts)
        res[f"dice_count_{suffix}"] = dice(t1_counts, t2_counts)
        res[f"cosine_count_{suffix}"] = cosine(t1_counts, t2_counts)
        res[f"jaccard_tfidf_{suffix}"] = jaccard(t1_tfidfs, t2_tfidfs)
        res[f"dice_tfidf_{suffix}"] = dice(t1_tfidfs, t2_tfidfs)
        res[f"cosine_tfidf_{suffix}"] = cosine(t1_tfidfs, t2_tfidfs)

        # Print similarity statistics
        for col in [f"jaccard_count_{suffix}", f"dice_count_{suffix}", f"cosine_count_{suffix}",
                    f"jaccard_tfidf_{suffix}", f"dice_tfidf_{suffix}", f"cosine_tfidf_{suffix}"]:
            print(f"\n{col} statistics:")
            print(f"- Mean: {res[col].mean():.4f}")
            print(f"- Std: {res[col].std():.4f}")
            print(f"- Min: {res[col].min():.4f}")
            print(f"- Max: {res[col].max():.4f}")

    return res


if __name__ == "__main__":
    # Define paths
    TRAIN_DF_PATH = r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_train-data_batch.csv"
    TEST_DF_PATH = r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_eval_data_202201.csv"
    TRAIN_TEXT_PATH = r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe.csv"
    TEST_TEXT_PATH = r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe_eval.csv"

    print("Loading datasets...")
    train = pd.read_csv(TRAIN_DF_PATH)
    test = pd.read_csv(TEST_DF_PATH)
    print(f"Initial shapes - Train: {train.shape}, Test: {test.shape}")

    print("\nLoading text data...")
    train_text = pd.read_csv(TRAIN_TEXT_PATH, low_memory=False)
    test_text = pd.read_csv(TEST_TEXT_PATH, low_memory=False)

    print("\nCleaning text data...")
    print(f"Train text before cleaning: {train_text.shape}")
    train_text = train_text.dropna(subset=["title", "text"]).reset_index(drop=True)
    print(f"Train text after cleaning: {train_text.shape}")

    print(f"Test text before cleaning: {test_text.shape}")
    test_text = test_text.dropna(subset=["title", "text"]).reset_index(drop=True)
    print(f"Test text after cleaning: {test_text.shape}")

    print("\nProcessing text IDs...")
    train_text["text_id"] = train_text["text_id"].astype(str)
    test_text["text_id"] = test_text["text_id"].astype(str)

    print("\nFiltering valid pairs...")
    print(f"Train pairs before filtering: {len(train)}")
    train = train[train["pair_id"].map(lambda x: contain_text(train_text, x))].reset_index(drop=True)
    print(f"Train pairs after filtering: {len(train)}")

    print(f"Test pairs before filtering: {len(test)}")
    test = test[test["pair_id"].map(lambda x: contain_text(test_text, x))].reset_index(drop=True)
    print(f"Test pairs after filtering: {len(test)}")

    print("\nExtracting text IDs from pairs...")
    for df in [train, test]:
        df["text_id1"] = df["pair_id"].str.split("_").map(lambda x: x[0])
        df["text_id2"] = df["pair_id"].str.split("_").map(lambda x: x[1])

    print("\nMerging text data...")
    train = pd.merge(
        train,
        train_text[["text_id", "title", "text"]],
        left_on="text_id1",
        right_on="text_id",
        how="left",
    )
    train = pd.merge(
        train,
        train_text[["text_id", "title", "text"]],
        left_on="text_id2",
        right_on="text_id",
        how="left",
        suffixes=("_1", "_2"),
    )

    test = pd.merge(
        test,
        test_text[["text_id", "title", "text"]],
        left_on="text_id1",
        right_on="text_id",
        how="left",
    )
    test = pd.merge(
        test,
        test_text[["text_id", "title", "text"]],
        left_on="text_id2",
        right_on="text_id",
        how="left",
        suffixes=("_1", "_2"),
    )

    print("\nCombining train and test for feature creation...")
    df = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(f"Combined shape: {df.shape}")

    print("\nCreating features...")
    df = create_match_features(df, ["title", "text"])

    use_cols = [
        "jaccard_count_title", "dice_count_title", "cosine_count_title",
        "jaccard_tfidf_title", "dice_tfidf_title", "cosine_tfidf_title",
        "jaccard_count_text", "dice_count_text", "cosine_count_text",
        "jaccard_tfidf_text", "dice_tfidf_text", "cosine_tfidf_text",
    ]

    print("\nSplitting features back into train and test...")
    X_train = df[use_cols][: len(train)].reset_index(drop=True)
    X_test = df[use_cols][len(train):].reset_index(drop=True)

    print(f"Final shapes - X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\nSaving features...")
    X_train.to_csv(r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\X_train.csv", index=False)
    X_test.to_csv(r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\X_test.csv", index=False)
    print("Processing completed!")