# src/model_trainer.py

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from src.utils import compute_metrics # Import the compute_metrics function

class BERTModelTrainer:
    """
    Encapsulates the BERT model loading, training configuration, and training process.
    """
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, id2label=None, label2id=None):
        """
        Initializes the model trainer.

        Args:
            model_name (str): The name of the pre-trained model to fine-tune.
            num_labels (int): The number of output labels for classification.
            id2label (dict, optional): A dictionary mapping label IDs to label names.
            label2id (dict, optional): A dictionary mapping label names to label IDs.
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id

        print(f"Loading model: {self.model_name} with {self.num_labels} labels...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # Unfreeze all the model parameters.
        # While parameters are typically unfrozen by default when loading for fine-tuning,
        # this explicitly ensures all parameters are trainable.
        for param in self.model.parameters():
            param.requires_grad = True

        # DataCollatorWithPadding dynamically pads batches to the longest sequence in that batch.
        # It needs the tokenizer to know the padding token ID.
        self.data_collator = DataCollatorWithPadding(tokenizer=None) # Tokenizer will be set by Trainer later

    def train(self, train_dataset, eval_dataset, output_dir="./results", epochs=2, batch_size=16, learning_rate=2e-5):
        """
        Configures and starts the model training process.

        Args:
            train_dataset (torch.utils.data.Dataset): The training dataset.
            eval_dataset (torch.utils.data.Dataset): The evaluation dataset.
            output_dir (str): Directory to save model checkpoints and logs.
            epochs (int): Number of training epochs.
            batch_size (int): Training batch size.
            learning_rate (float): Learning rate for the optimizer.
        """
        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir="./data/spam_not_spam",
            # Set the learning rate
            learning_rate=2e-5,
            # Set the per device train batch size and eval batch size
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            # Evaluate and save the model after each epoch
            evaluation_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,
            weight_decay=0.01,
            load_best_model_at_end=True,
        )

        print("Initializing Trainer...")
        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics, # Function to compute metrics during evaluation
            data_collator=self.data_collator, # Data collator for dynamic padding of batches
        )

        print("Starting training...")
        # Train the model
        trainer.train()

        print("Training complete. Evaluating model...")
        # Evaluate the model after training
        results = trainer.evaluate()
        print(f"Evaluation results: {results}")

        print(f"Saving fine-tuned model to {output_dir}/final_model")
        # Save the final fine-tuned model
        self.model.save_pretrained(f"{output_dir}/final_model")
        # The tokenizer is saved separately in main.py, as it's managed by data_processor

        return results, trainer # Return trainer instance for post-training prediction

