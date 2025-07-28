# main.py

import os
import pandas as pd
from src.data_processor import SMSDataProcessor
from src.model_trainer import BERTModelTrainer

def main():
    """
    Main function to run the SMS Spam classification fine-tuning process.
    """
    # Define constants for configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 128
    NUM_LABELS = 2 # 'ham' and 'spam'
    OUTPUT_DIR = "./spam_classifier_results"
    EPOCHS = 2 # Number of training epochs, as per your notebook
    BATCH_SIZE = 16 # Batch size for training and evaluation
    LEARNING_RATE = 2e-5 # Learning rate for the optimizer
    TEST_SPLIT_SIZE = 0.2 # Proportion of data to use for the test set
    RANDOM_SEED = 23 # Seed for reproducibility of the data split

    # Define label mappings for the model and display
    id2label = {0: "not spam", 1: "spam"}
    label2id = {"not spam": 0, "spam": 1}

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("--- Starting SMS Spam Classification Fine-tuning ---")

    # Step 1: Process Data
    # Initialize the data processor with the model name and max sequence length
    data_processor = SMSDataProcessor(model_name=MODEL_NAME, max_length=MAX_LENGTH)
    # Load and process the dataset. This includes tokenization, label mapping,
    # and creating the train-test split from the original 'train' split.
    processed_dataset = data_processor.load_and_process_data(test_size=TEST_SPLIT_SIZE, seed=RANDOM_SEED)

    # Access the newly created 'train' and 'test' splits from the processed dataset
    train_dataset = processed_dataset["train"]
    eval_dataset = processed_dataset["test"]

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # Step 2: Train Model
    # Initialize the model trainer with the model name, number of labels, and label mappings
    model_trainer = BERTModelTrainer(
        model_name=MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label, # Pass id2label to the model trainer
        label2id=label2id  # Pass label2id to the model trainer
    )
    # Set the tokenizer for the data collator within the model trainer.
    # This is crucial for dynamic padding during training.
    model_trainer.data_collator.tokenizer = data_processor.tokenizer

    # Start the training process. The train method returns evaluation results and the trainer instance.
    _, trainer_instance = model_trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )

    print("--- Fine-tuning process completed successfully! ---")
    # Save the tokenizer alongside the model for easy loading later
    data_processor.tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")
    print(f"Model and tokenizer saved to: {os.path.abspath(OUTPUT_DIR)}/final_model")

    # Step 3: Post-training Prediction and Review (as in your original notebook)
    print("\n--- Performing post-training prediction for manual review ---")

    # Select specific items from the evaluation dataset for detailed review
    items_for_manual_review = eval_dataset.select(
        [0, 1, 22, 31, 43, 292, 448, 487]
    )

    # Perform predictions on the selected items using the trained model
    results = trainer_instance.predict(items_for_manual_review)

    # Map numerical predictions and true labels back to their string representations
    predicted_labels = [id2label[p] for p in results.predictions.argmax(axis=1)]
    true_labels = [id2label[l] for l in results.label_ids]

    # Create a pandas DataFrame to display the original SMS, predicted label, and true label
    df = pd.DataFrame(
        {
            "sms": [item["sms"] for item in items_for_manual_review], # 'sms' column is retained in data_processor
            "predictions": predicted_labels,
            "labels": true_labels,
        }
    )
    # Set pandas option to display full content of SMS messages
    pd.set_option("display.max_colwidth", None)
    print(df)

if __name__ == "__main__":
    main()

