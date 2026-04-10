"""
Advanced Model Training with Multiple Algorithms
Includes: Neural Networks, Voting Ensembles, Stacking
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Only import TensorFlow if available
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Neural Network will be skipped.")

from sklearn.ensemble import VotingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score


class NeuralNetworkFraudDetector:
    """
    Deep Learning model for fraud detection
    """
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None
        if TENSORFLOW_AVAILABLE:
            self.model = self.build_model()
    
    def build_model(self):
        """
        Build neural network architecture
        """
        if not TENSORFLOW_AVAILABLE:
            return None
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # Hidden layers with batch normalization
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with class weight for imbalance
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """
        Train neural network with early stopping
        """
        if self.model is None:
            print("⚠️ TensorFlow not available. Returning None.")
            return None
        
        # Calculate class weights to handle imbalance
        class_weight = {
            0: 1,
            1: len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        }
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return history
    
    def predict_proba(self, X):
        """
        Get probability predictions
        """
        if self.model is None:
            return None
        
        predictions = self.model.predict(X, verbose=0)
        return np.column_stack([1 - predictions, predictions])


class EnsembleModels:
    """
    Advanced ensemble methods
    """
    
    @staticmethod
    def create_voting_ensemble(models_dict):
        """
        Create voting classifier ensemble
        
        Args:
            models_dict: Dictionary of {name: model}
        """
        estimators = [(name, model) for name, model in models_dict.items()]
        
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability predictions
            weights=[1, 1, 1, 1, 1]  # Equal weights
        )
        
        return voting_clf
    
    @staticmethod
    def create_stacking_ensemble(base_models, meta_model):
        """
        Create stacking classifier
        
        Args:
            base_models: List of base models
            meta_model: Final meta-learner
        """
        stacking_clf = StackingClassifier(
            estimators=[
                ('lr', base_models[0]),
                ('rf', base_models[1]),
                ('xgb', base_models[2]),
                ('lgb', base_models[3])
            ],
            final_estimator=meta_model,
            cv=5
        )
        
        return stacking_clf


class CostSensitiveLearning:
    """
    Cost-sensitive learning for fraud detection
    """
    
    @staticmethod
    def calculate_class_weights(y):
        """
        Calculate class weights for imbalanced data
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        weights = {}
        
        for class_val in np.unique(y):
            weights[class_val] = n_samples / (n_classes * np.sum(y == class_val))
        
        return weights
    
    @staticmethod
    def create_cost_sensitive_model(base_model, y_train):
        """
        Create cost-sensitive version of model
        """
        class_weights = CostSensitiveLearning.calculate_class_weights(y_train)
        
        # Apply class weights to model
        if hasattr(base_model, 'class_weight'):
            base_model.class_weight = class_weights
        elif hasattr(base_model, 'sample_weight'):
            # Handle differently for models that need sample weights
            pass
        
        return base_model, class_weights