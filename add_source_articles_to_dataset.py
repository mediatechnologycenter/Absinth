from datasets import load_dataset
import pandas as pd
from typing import Dict


def read_source_articles(source_articles_file_path: str) -> Dict:
    merged_dict = {}

    # Open the JSONL file and iterate through its lines
    with open(source_articles_file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line)
            merged_dict.update(json_obj)
        return merged_dict


def main():
    source_articles_file_path = "path/to/absinth_source_articles.jsonl"
    absinth_dataset_hf_id = "mtc/absinth_german_faithfulness_detection_dataset"
    source_articles_dict = read_source_articles(source_articles_file_path)
    absinth_dataset = load_dataset(absinth_dataset_hf_id)

    def add_source_article(example):
        # The source articles are a concatenation of the article's lead text and main text separated by a newline character
        source_article_column_name = "lead_with_article"
        article_id = example["article_id"]
        example[source_article_column_name] = source_articles_dict[article_id]
        return example

    splits = ["train", "validation", "test"]
    for split in splits:
        absinth_dataset[split] = absinth_dataset[split].map(add_source_article)

    # Perform further processing with absinth dataset...


if __name__ == "__main__":
    main()
