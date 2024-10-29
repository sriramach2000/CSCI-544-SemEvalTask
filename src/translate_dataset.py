import pandas as pd
import time
from googletrans import Translator
from tqdm import tqdm
import numpy as np


def create_translator():
    """Create a translator instance with retry mechanism."""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            return Translator()
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Failed to create translator after {max_attempts} attempts: {e}")
                raise
            time.sleep(1)


def safe_translate(text, translator, source_lang=None):
    """Safely translate text with error handling and retries."""
    if pd.isna(text):
        return text

    # If text is empty or just whitespace, return as is
    if not str(text).strip():
        return text

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # Add delay between retries to avoid API limits
            if attempt > 0:
                time.sleep(2)

            result = translator.translate(str(text), dest='en', src=source_lang)
            return result.text
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Translation failed after {max_attempts} attempts for text: {text[:100]}...")
                return text
            continue


def merge_df_and_text(df, text_dataframe):
    """Merge dataframes with improved error handling."""
    if text_dataframe is None or text_dataframe.empty:
        print("Warning: Text dataframe is empty or None")
        return df

    # Handle missing columns gracefully
    required_cols = ["title", "text", "text_id"]
    missing_cols = [col for col in required_cols if col not in text_dataframe.columns]
    if missing_cols:
        print(f"Warning: Missing columns in text_dataframe: {missing_cols}")
        for col in missing_cols:
            text_dataframe[col] = np.nan

    # Clean and prepare text dataframe
    text_dataframe = text_dataframe.copy()
    text_dataframe["title"] = text_dataframe["title"].fillna("")
    text_dataframe["text"] = text_dataframe["text"].fillna("")
    text_dataframe["title"] = text_dataframe["title"] + "[SEP]" + text_dataframe["text"]
    text_dataframe["text_id"] = text_dataframe["text_id"].astype(str)

    # Filter valid pairs
    df = df[df["pair_id"].map(lambda x: contain_text(text_dataframe, x))].reset_index(drop=True)

    # Split pair_id into text_ids
    df["text_id1"] = df["pair_id"].str.split("_").str[0]
    df["text_id2"] = df["pair_id"].str.split("_").str[1]

    # Perform merges with error handling
    for suffix, id_col in [("_1", "text_id1"), ("_2", "text_id2")]:
        try:
            df = pd.merge(
                df,
                text_dataframe[["text_id", "title"]],
                left_on=id_col,
                right_on="text_id",
                how="left",
                suffixes=("", suffix) if suffix == "_1" else ("_1", "_2")
            )
        except Exception as e:
            print(f"Merge failed for {id_col}: {str(e)}")
            df[f"title{suffix}"] = np.nan

    return df


def contain_text(text_df, pair_id: str):
    """Check if both texts in a pair exist in the dataframe."""
    try:
        text_id1, text_id2 = pair_id.split("_")
        return (text_id1 in text_df.text_id.values) and (text_id2 in text_df.text_id.values)
    except:
        return False


def translation(df, col):
    """Translate text with improved error handling and progress tracking."""
    if df is None or df.empty:
        print(f"Warning: Empty dataframe provided for translation of column {col}")
        return df

    trans_texts = []
    error_count = 0
    skipped_count = 0
    translated_count = 0

    # Create translator instance
    translator = create_translator()
    res = df.copy()

    # Determine language column
    lang_col = "url1_lang" if "1" in col else "url2_lang"
    if lang_col not in df.columns:
        print(f"Warning: Language column {lang_col} not found. Assuming non-English text.")
        df[lang_col] = 'non-en'

    print(f"\nTranslating column: {col}")
    for text, lang in tqdm(zip(df[col], df[lang_col]), total=len(df), desc=f"Translating {col}"):
        try:
            if pd.isna(text) or pd.isna(lang):
                trans_texts.append(text)
                skipped_count += 1
                continue

            if lang == "en":
                trans_texts.append(text)
                skipped_count += 1
            else:
                translated_text = safe_translate(text, translator, source_lang=lang if lang != 'non-en' else None)
                trans_texts.append(translated_text)
                translated_count += 1

        except Exception as e:
            error_count += 1
            trans_texts.append(text)
            print(f"\nTranslation error for text: {text[:100]}... Error: {str(e)}")

    print(f"\nTranslation Summary for {col}:")
    print(f"- Total processed: {len(trans_texts)}")
    print(f"- Successfully translated: {translated_count}")
    print(f"- Skipped (English/Empty): {skipped_count}")
    print(f"- Errors: {error_count}")

    res[col] = trans_texts
    return res


if __name__ == "__main__":
    try:
        # Load dataframes with error handling
        print("Loading input files...")
        train = pd.read_csv(
            r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_train-data_batch.csv")
        test = pd.read_csv(
            r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_eval_data_202201.csv")

        try:
            text_dataframe = pd.read_csv(r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe.csv",
                                         low_memory=False)
        except Exception as e:
            print(f"Warning: Could not load training text dataframe: {e}")
            text_dataframe = None

        try:
            text_dataframe_eval = pd.read_csv(
                r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe_eval.csv", low_memory=False)
        except Exception as e:
            print(f"Warning: Could not load eval text dataframe: {e}")
            text_dataframe_eval = None

        # Process training data
        print("\nProcessing training data...")
        train = merge_df_and_text(train, text_dataframe)

        # Process test data
        print("\nProcessing test data...")
        test = merge_df_and_text(test, text_dataframe_eval)

        print(f"\nDataset shapes after merging:")
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")

        # Translate both datasets
        print("\nTranslating training data...")
        train = translation(train, "title_1")
        train = translation(train, "title_2")

        print("\nTranslating test data...")
        test = translation(test, "title_1")
        test = translation(test, "title_2")

        # Save results
        print("\nSaving processed datasets...")
        train.to_csv(r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\train.csv", index=False)
        test.to_csv(r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\test.csv", index=False)

        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during processing: {str(e)}")