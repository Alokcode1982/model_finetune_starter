# src/data_processor.py

import os
import shutil  # Import shutil for directory removal
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HOME  # Updated import for cache directory constant


class SMSDataProcessor:
    """
    Handles loading, tokenizing, and preparing the SMS Spam dataset.
    """

    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        """
        Initializes the data processor with a tokenizer.

        Args:
            model_name (str): The name of the pre-trained model for the tokenizer.
            max_length (int): The maximum sequence length for tokenization.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        # Define label mappings for consistency
        self.id2label = {0: "not spam", 1: "spam"}
        self.label2id = {"not spam": 0, "spam": 1}

    def load_and_process_data(self, test_size=0.2, seed=23):
        """
        Loads the SMS Spam dataset, maps labels, tokenizes the text,
        and creates a train-test split.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            seed (int): Random seed for reproducibility of the split.

        Returns:
            datasets.DatasetDict: The processed dataset with train and test splits.
        """
        print("Loading SMS Spam dataset...")

        # --- Start of Cache Clearing Logic ---
        # Get the default cache directory for datasets using the updated constant
        cache_dir = HF_HOME  # Use HF_HOME constant for the base cache directory
        dataset_cache_path = os.path.join(cache_dir, "datasets", "sms_spam")

        # Check if the cache directory exists and remove it
        if os.path.exists(dataset_cache_path):
            print(f"Clearing existing cache for sms_spam dataset at: {dataset_cache_path}")
            try:
                shutil.rmtree(dataset_cache_path)
                print("Cache cleared successfully.")
            except OSError as e:
                print(f"Error clearing cache: {e}. Attempting to proceed without clearing.")
        else:
            print(f"No existing cache found for sms_spam at: {dataset_cache_path}")
        # --- End of Cache Clearing Logic ---

        # Load the dataset from the 'datasets' library
        # The sms_spam dataset only has a 'train' split, so we explicitly load it
        # and then split it into train and test.
        # download_mode="force_redownload" is kept for an extra layer of safety.
        dataset = load_dataset("sms_spam", split="train", download_mode="force_redownload")

        # Create the train-test split immediately after loading
        dataset_splits = dataset.train_test_split(test_size=test_size, shuffle=True, seed=seed)

        # The dataset has 'label' as 'ham' or 'spam'. Map them to 0 and 1.
        # 'ham' -> 0, 'spam' -> 1
        def map_labels(example):
            example['label'] = self.label2id['spam'] if example['label'] == 'spam' else self.label2id['not spam']
            return example

        print("Mapping labels...")
        # Apply the label mapping to both train and test splits
        dataset_splits = dataset_splits.map(map_labels)

        # Tokenize the text messages
        def tokenize_function(examples):
            # Ensure 'sms' column is used for tokenization
            return self.tokenizer(examples["sms"], truncation=True, max_length=self.max_length)

        print("Tokenizing dataset...")
        # Apply the tokenization function to both train and test splits
        tokenized_dataset = dataset_splits.map(tokenize_function, batched=True)

        # Do NOT remove the original 'sms' column here, as it's needed for final display.
        # The Trainer will automatically ignore columns not in the model's forward pass.

        # Rename 'label' to 'labels' to match the Trainer's expected input
        tokenized_dataset = tokenized_dataset.rename_columns({"label": "labels"})

        # Set the format to PyTorch tensors
        tokenized_dataset.set_format("torch")

        return tokenized_dataset

