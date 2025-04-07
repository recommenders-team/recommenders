# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_TIMESTAMP_COL,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingRankerDataset(Dataset):
    """PyTorch Dataset for the Embedding Ranker model.
    
    This dataset prepares user-item interaction data for training the model.
    """
    
    def __init__(self, df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, 
                 col_rating=DEFAULT_RATING_COL):
        """Initialize dataset.
        
        Args:
            df (pandas.DataFrame): Dataframe containing the user-item interactions.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
        """
        self.df = df
        self.users = df[col_user].values
        self.items = df[col_item].values
        self.ratings = df[col_rating].values.astype(np.float32)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.users[idx],
            'item_id': self.items[idx],
            'rating': self.ratings[idx]
        }


class RankingModel(nn.Module):
    """PyTorch implementation of a basic embedding-based neural ranking model.
    
    This model creates embedding layers for users and items, and predicts ratings
    through a combination of dot product and fully connected layers.
    
    Similar to TFRS BasicModel (https://www.tensorflow.org/recommenders/examples/basic_ranking)
    but implemented in PyTorch.
    """
    
    def __init__(self, num_users, num_items, embedding_dim=32, hidden_units=[64, 32]):
        """Initialize model.
        
        Args:
            num_users (int): Number of unique users in the dataset.
            num_items (int): Number of unique items in the dataset.
            embedding_dim (int): Dimension of the embeddings.
            hidden_units (list): List of hidden unit sizes for each layer.
        """
        super(RankingModel, self).__init__()
        
        # Create embedding layers for users and items
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        
        # Initialize embeddings with normal distribution
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
        # Define the layers of the rating prediction network
        layers = []
        
        # Input is the concatenated embeddings (2 * embedding_dim) plus dot product (1)
        input_dim = 2 * embedding_dim + 1
        
        # Create hidden layers with ReLU activation
        for units in hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            input_dim = units
        
        # Output layer (predicts a single rating)
        layers.append(nn.Linear(input_dim, 1))
        
        # Create the sequential model
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, user_id, item_id):
        """Forward pass through the model.
        
        Args:
            user_id (torch.Tensor): Tensor of user IDs.
            item_id (torch.Tensor): Tensor of item IDs.
            
        Returns:
            torch.Tensor: Predicted ratings.
        """
        # Get embeddings for the batch
        user_embeds = self.user_embedding(user_id)
        item_embeds = self.item_embedding(item_id)
        
        # Calculate the dot product
        dot_product = torch.sum(user_embeds * item_embeds, dim=1, keepdim=True)
        
        # Concatenate embeddings and dot product
        concat = torch.cat([user_embeds, item_embeds, dot_product], dim=1)
        
        # Pass through the prediction layers
        prediction = self.predictor(concat)
        
        return prediction.squeeze()


class NNEmbeddingRanker:
    """Neural Network Embedding Ranker model for rating prediction and item recommendation.
    
    This class provides methods for training an embedding-based ranking model and making
    predictions using PyTorch.
    """
    
    def __init__(
        self,
        n_factors=32,
        hidden_units=[64, 32],
        n_epochs=10,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0,
        num_workers=0,
        use_cuda=True,
        seed=None,
        verbose=False,
    ):
        """Initialize model parameters.
        
        Args:
            n_factors (int): Dimension of the embedding vectors.
            hidden_units (list): List of hidden unit sizes for prediction layers.
            n_epochs (int): Number of epochs for training.
            batch_size (int): Number of examples per training batch.
            learning_rate (float): Learning rate for the optimizer.
            weight_decay (float): L2 regularization strength.
            num_workers (int): Number of workers for data loading.
            use_cuda (bool): Whether to use GPU if available.
            seed (int): Random seed for reproducibility.
            verbose (bool): Whether to show training progress.
        """
        self.n_factors = n_factors
        self.hidden_units = hidden_units
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_workers = num_workers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.model = None
        
        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if self.use_cuda:
                torch.cuda.manual_seed_all(seed)
                
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def fit(self, train_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, 
            col_rating=DEFAULT_RATING_COL, val_df=None):
        """Train the embedding ranker model.
        
        Args:
            train_df (pandas.DataFrame): Training data.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            val_df (pandas.DataFrame, optional): Validation data.
            
        Returns:
            NNEmbeddingRanker: Trained model instance.
        """
        # Create mapping of user/item IDs to internal indices
        self.user_id_map = {user_id: i for i, user_id in enumerate(train_df[col_user].unique())}
        self.item_id_map = {item_id: i for i, item_id in enumerate(train_df[col_item].unique())}
        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)
        
        logger.info(f"Number of unique users: {self.n_users}")
        logger.info(f"Number of unique items: {self.n_items}")
        
        # Create mapped dataframes
        train_mapped_df = train_df.copy()
        train_mapped_df[col_user] = train_mapped_df[col_user].map(self.user_id_map)
        train_mapped_df[col_item] = train_mapped_df[col_item].map(self.item_id_map)
        
        # Create training dataset and dataloader
        train_dataset = EmbeddingRankerDataset(train_mapped_df, col_user, col_item, col_rating)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.use_cuda,
        )
        
        # Validation data, if provided
        if val_df is not None:
            val_mapped_df = val_df.copy()
            # Skip users/items not in training
            val_mapped_df = val_mapped_df[val_mapped_df[col_user].isin(self.user_id_map.keys())]
            val_mapped_df = val_mapped_df[val_mapped_df[col_item].isin(self.item_id_map.keys())]
            val_mapped_df[col_user] = val_mapped_df[col_user].map(self.user_id_map)
            val_mapped_df[col_item] = val_mapped_df[col_item].map(self.item_id_map)
            
            val_dataset = EmbeddingRankerDataset(val_mapped_df, col_user, col_item, col_rating)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.use_cuda,
            )
        
        # Initialize model
        self.model = RankingModel(
            num_users=self.n_users,
            num_items=self.n_items,
            embedding_dim=self.n_factors,
            hidden_units=self.hidden_units,
        )
        self.model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss = 0.0
            
            # Progress bar if verbose
            data_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs}") if self.verbose else train_loader
            
            for batch in data_iterator:
                # Get data
                user_id = batch['user_id'].to(self.device)
                item_id = batch['item_id'].to(self.device)
                rating = batch['rating'].to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                prediction = self.model(user_id, item_id)
                
                # Compute loss
                loss = criterion(prediction, rating)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * user_id.size(0)
            
            # Calculate average loss
            train_loss = running_loss / len(train_dataset)
            val_loss = None
            
            # Validation, if provided
            if val_df is not None:
                val_loss = self._evaluate(val_loader, criterion)
                
            # Print progress
            if self.verbose:
                if val_loss:
                    logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{self.n_epochs}, Train Loss: {train_loss:.4f}")
        
        return self
    
    def _evaluate(self, dataloader, criterion):
        """Evaluate the model on the given dataloader.
        
        Args:
            dataloader (DataLoader): DataLoader containing evaluation data.
            criterion: Loss function.
            
        Returns:
            float: Average loss.
        """
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for batch in dataloader:
                user_id = batch['user_id'].to(self.device)
                item_id = batch['item_id'].to(self.device)
                rating = batch['rating'].to(self.device)
                
                prediction = self.model(user_id, item_id)
                loss = criterion(prediction, rating)
                
                running_loss += loss.item() * user_id.size(0)
        
        return running_loss / len(dataloader.dataset)
    
    def predict(self, user_id, item_id, is_list=False):
        """Predict ratings for given user-item pairs.
        
        Args:
            user_id (int, list): User ID(s).
            item_id (int, list): Item ID(s).
            is_list (bool): Whether input is a list.
            
        Returns:
            float, list: Predicted rating(s).
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        if is_list:
            # Convert lists to tensors
            user_indices = [self.user_id_map.get(u, 0) for u in user_id]
            item_indices = [self.item_id_map.get(i, 0) for i in item_id]
            user_tensor = torch.tensor(user_indices, device=self.device)
            item_tensor = torch.tensor(item_indices, device=self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(user_tensor, item_tensor).cpu().numpy()
            
            return predictions
        else:
            # Single prediction
            user_idx = self.user_id_map.get(user_id, 0)
            item_idx = self.item_id_map.get(item_id, 0)
            
            user_tensor = torch.tensor([user_idx], device=self.device)
            item_tensor = torch.tensor([item_idx], device=self.device)
            
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(user_tensor, item_tensor).cpu().numpy()[0]
            
            return prediction
    
    def recommend_k_items(
        self, test_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, 
        col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL,
        top_k=10, remove_seen=True
    ):
        """Generate top-k recommendations for each user in the test set.
        
        Args:
            test_df (pandas.DataFrame): Test data.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name.
            col_prediction (str): Prediction column name.
            top_k (int): Number of top items to recommend.
            remove_seen (bool): Whether to remove items that appear in the training data.
            
        Returns:
            pandas.DataFrame: DataFrame with user-item-prediction columns sorted by prediction in descending order.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Get all users in test set
        test_users = test_df[col_user].unique()
        
        # Filter users that are not in the training set
        test_users = [user for user in test_users if user in self.user_id_map]
        
        if not test_users:
            raise ValueError("No valid users found in test set")
        
        # Get all items
        all_items = list(self.item_id_map.keys())
        
        # Generate all possible user-item pairs for prediction
        user_item_pairs = []
        users = []
        items = []
        for user in test_users:
            for item in all_items:
                users.append(user)
                items.append(item)
        
        # Get predictions for all pairs
        predictions = self.predict(users, items, is_list=True)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            col_user: users,
            col_item: items,
            col_prediction: predictions
        })
        
        # If remove_seen is True, remove items that appear in the training data
        if remove_seen:
            # Create a set of user-item pairs from the training data
            # This is based on the user and item IDs we mapped during training
            seen_pairs = set((u, i) for u, i in zip(self.user_id_map.keys(), self.item_id_map.keys()))
            
            # Filter out seen pairs
            result_df = result_df[~result_df.apply(lambda row: (row[col_user], row[col_item]) in seen_pairs, axis=1)]
        
        # Get top-k recommendations for each user
        top_k_df = (
            result_df
            .sort_values([col_user, col_prediction], ascending=[True, False])
            .groupby(col_user).head(top_k)
            .reset_index(drop=True)
        )
        
        return top_k_df
    
    def save(self, filepath):
        """Save model to a file.
        
        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Prepare model data for saving
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'n_factors': self.n_factors,
            'hidden_units': self.hidden_units,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map,
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, use_cuda=True):
        """Load a saved model.
        
        Args:
            filepath (str): Path to the saved model.
            use_cuda (bool): Whether to use GPU if available.
            
        Returns:
            NNEmbeddingRanker: Loaded model instance.
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the model data
        map_location = None if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        model_data = torch.load(filepath, map_location=map_location)
        
        # Create a new instance
        instance = cls(
            n_factors=model_data['n_factors'],
            hidden_units=model_data['hidden_units'],
            use_cuda=use_cuda,
        )
        
        # Initialize the model
        instance.n_users = model_data['n_users']
        instance.n_items = model_data['n_items']
        instance.user_id_map = model_data['user_id_map']
        instance.item_id_map = model_data['item_id_map']
        
        # Create and load the model
        instance.model = RankingModel(
            num_users=instance.n_users,
            num_items=instance.n_items,
            embedding_dim=instance.n_factors,
            hidden_units=instance.hidden_units,
        )
        instance.model.load_state_dict(model_data['model_state_dict'])
        instance.model.to(instance.device)
        instance.model.eval()
        
        logger.info(f"Model loaded from {filepath}")
        return instance 