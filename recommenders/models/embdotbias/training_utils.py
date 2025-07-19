# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import torch
import torch.optim as optim
from torch.nn import MSELoss
import logging

from recommenders.utils.constants import (
    DEFAULT_USER_COL as USER,
    DEFAULT_ITEM_COL as ITEM,
    DEFAULT_RATING_COL as RATING,
    DEFAULT_TIMESTAMP_COL as TIMESTAMP,
    DEFAULT_PREDICTION_COL as PREDICTION,
)

# Set up logger
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, learning_rate=1e-3, weight_decay=0.01):
        """
        Initializes the RecommenderTrainer.

        Args:
            model: The PyTorch model to train.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.99),
            eps=1e-5,
            weight_decay=weight_decay,
        )
        self.loss_fn = MSELoss()

    def train_epoch(self, train_dl):
        """
        Trains the model for one epoch.

        Args:
            train_dl: The training data loader.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in train_dl:
            users_items, ratings = batch
            users_items = users_items.to(self.device)
            ratings = ratings.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(users_items)
            loss = self.loss_fn(predictions.view(-1), ratings.view(-1))
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(train_dl)

    def validate(self, valid_dl):
        """
        Validates the model on the validation set.

        Args:
            valid_dl: The validation data loader.

        Returns:
            float or None: The average validation loss, or None if the validation data loader is empty.
        """
        self.model.eval()
        total_loss = 0
        try:
            with torch.no_grad():
                for batch in valid_dl:
                    users_items, ratings = batch
                    users_items = users_items.to(self.device)
                    ratings = ratings.to(self.device)

                    predictions = self.model(users_items)
                    loss = self.loss_fn(predictions.view(-1), ratings.view(-1))
                    total_loss += loss.item()
            return total_loss / len(valid_dl)
        except ZeroDivisionError:
            return None

    def fit(self, train_dl, valid_dl, n_epochs):
        """
        Trains the model for a specified number of epochs.

        Args:
            train_dl: The training data loader.
            valid_dl: The validation data loader.
            n_epochs (int): The number of epochs to train for.
        """
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_dl)
            valid_loss = self.validate(valid_dl)
            logger.info(f"Epoch {epoch+1}/{n_epochs}:")
            logger.info(f"Train Loss: {train_loss}")
            logger.info(f"Valid Loss: {valid_loss}")


def predict_rating(model, user_id, item_id):
    """
    Predicts the rating for a given user and item.

    Args:
        user_id (str): The ID of the user.
        item_id (str): The ID of the item.

    Returns:
        float or None: The predicted rating, or None if an error occurs.
    """
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Ensure model is on the same device as input

    with torch.no_grad():
        try:
            user_idx = model._get_idx([user_id], is_item=False)
            item_idx = model._get_idx([item_id], is_item=True)

            x = torch.stack([user_idx, item_idx], dim=1).to(device)

            pred = model(x)
            return pred.item()
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None
