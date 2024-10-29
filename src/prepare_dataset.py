import json
import pathlib
from typing import List, Set, Optional
import os

import pandas as pd
from tqdm import tqdm


def fetch_dataframe(text_id: str, meta, allow_missing: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch and process a single JSON file into a dataframe.
    If allow_missing=True, returns placeholder for missing files.
    """
    search_results = list(meta.glob(f"*/{text_id}.json"))
    if not search_results:
        if allow_missing:
            print(f"\nMissing metadata for {text_id} - creating placeholder")
            return pd.DataFrame({
                'text_id': [text_id],
                'metadata_available': [False],
                'status': ['missing']
            })
        else:
            raise IndexError(f"No file found for text_id: {text_id}")

    try:
        text_path = str(search_results[0])
        with open(text_path, 'r', encoding='utf-8') as f:
            text_json = json.loads(f.read())

        res_df = pd.json_normalize(text_json)
        res_df["text_id"] = text_id
        res_df["metadata_available"] = True
        res_df["status"] = "complete"
        return res_df
    except Exception as e:
        print(f"Error processing file {text_id}: {str(e)}")
        if allow_missing:
            return pd.DataFrame({
                'text_id': [text_id],
                'metadata_available': [False],
                'status': ['error'],
                'error_message': [str(e)]
            })
        raise


def process_data(input_file: str, meta_path: str, output_file: str, is_eval: bool = False) -> None:
    """Process either training or evaluation data and save to separate files."""
    phase_name = "EVAL" if is_eval else "TRAIN"
    print(f"\nProcessing {phase_name} data:")

    # Verify input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load input CSV
    print(f"Loading input CSV from: {input_file}")
    data = pd.read_csv(input_file)
    print(f"Loaded CSV with shape: {data.shape}")

    # Extract text IDs
    text_ids: Set[str] = set(sum(data["pair_id"].str.split("_"), []))
    print(f"Found {len(text_ids)} unique text IDs to process")

    # Initialize metadata path
    meta = pathlib.Path(meta_path)
    if not meta.exists():
        print(f"Warning: Metadata directory {meta_path} does not exist")
        if not is_eval:  # Only raise error for training data
            raise ValueError(f"Metadata directory not found: {meta_path}")

    # Process files
    text_dfs: List[pd.DataFrame] = []
    successful = 0
    failed = 0
    missing = 0

    with tqdm(total=len(text_ids), desc=f"{phase_name} Progress") as pbar:
        for text_id in text_ids:
            try:
                res = fetch_dataframe(text_id, meta, allow_missing=is_eval)
                if res is not None:
                    if res['status'].iloc[0] == 'missing':
                        missing += 1
                    elif res['status'].iloc[0] == 'complete':
                        successful += 1
                    text_dfs.append(res)

            except Exception as e:
                failed += 1
                print(f"\nError processing {text_id}: {str(e)}")

            finally:
                pbar.update(1)
                total = successful + failed + missing
                if total > 0:
                    pbar.set_postfix({
                        'Complete': successful,
                        'Missing': missing,
                        'Failed': failed,
                        'Success Rate': f"{(successful / total * 100):.1f}%"
                    })

    if not text_dfs:
        print(f"\nNo data frames were created for {phase_name}. Check the errors above.")
        return

    # Combine results
    print(f"\nCombining processed {phase_name} dataframes...")
    text_df = pd.concat(text_dfs).reset_index(drop=True)

    # Add pair information
    pair_info = []
    for _, row in data.iterrows():
        id1, id2 = row['pair_id'].split('_')
        pair_info.extend([
            {'text_id': id1, 'pair_id': row['pair_id'], 'is_first': True},
            {'text_id': id2, 'pair_id': row['pair_id'], 'is_first': False}
        ])
    pair_df = pd.DataFrame(pair_info)

    # Merge with original data
    final_df = pd.merge(text_df, pair_df, on='text_id', how='left')
    print(f"Final {phase_name} DataFrame shape: {final_df.shape}")

    # Save results
    print(f"Saving {phase_name} dataframe to: {output_file}")
    final_df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n{phase_name} Summary:")
    print(f"- Total files processed: {successful + failed + missing}")
    print(f"- Successfully processed: {successful}")
    print(f"- Missing metadata: {missing}")
    print(f"- Failed to process: {failed}")
    print(f"- Success rate: {(successful / (successful + failed + missing) * 100):.1f}%")
    print(f"- Output saved to: {output_file}")
    print(f"- Final DataFrame shape: {final_df.shape}")

    # Data quality report
    print(f"\n{phase_name} Data Quality Report:")
    print(f"- Records with complete metadata: {(final_df['metadata_available'] == True).sum()}")
    print(f"- Records with missing metadata: {(final_df['metadata_available'] == False).sum()}")
    print(f"- Unique pair_ids: {final_df['pair_id'].nunique()}")


if __name__ == "__main__":
    # Process training data
    try:
        process_data(
            input_file=r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_train-data_batch.csv",
            meta_path=r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\output_dir",
            output_file=r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe.csv",
            is_eval=False
        )
    except Exception as e:
        print(f"Error in training phase: {str(e)}")

    # Process evaluation data
    try:
        process_data(
            input_file=r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\semeval-2022_task8_eval_data_202201.csv",
            meta_path=r"C:\Users\achar\PycharmProjects\semevaltask\input\semeval2022\output_dir",
            output_file=r"C:\Users\achar\PycharmProjects\semevaltask\input\text_dataframe_eval.csv",
            is_eval=True
        )
    except Exception as e:
        print(f"Error in evaluation phase: {str(e)}")