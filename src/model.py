import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve, precision_recall_curve, 
                             f1_score, precision_score, recall_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, SMOTENC
import joblib
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, Literal
import logging
import importlib
import pipeline
importlib.reload(pipeline)
from pipeline import *
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLModelTrainer:
    """
    Production-ready ML Model Trainer supporting multiple models.
    Works with both raw and preprocessed data.
    Supports XGBoost, Random Forest, and Logistic Regression with SMOTE.
    """
    
    def __init__(
        self,
        model_type: Literal['xgboost', 'random_forest', 'logistic_regression'] = 'xgboost',
        use_smote: bool = True,
        test_size: float = 0.2,
        random_state: int = 42,
        data_type: Literal['raw', 'preprocessed'] = 'preprocessed'
    ):
        """
        Initialize ML Model Trainer
        
        Parameters:
        -----------
        model_type : str
            Type of model: 'xgboost', 'random_forest', 'logistic_regression'
        use_smote : bool
            Whether to apply SMOTE for class balancing
        test_size : float
            Proportion of data for test set
        random_state : int
            Random state for reproducibility
        data_type : str
            'raw' or 'preprocessed' - determines SMOTE strategy
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.test_size = test_size
        self.random_state = random_state
        self.data_type = data_type
        
        self.model = None
        self.scaler = None
        self.categorical_indices = None
        self.categorical_columns = None
        self.feature_names = None
        
    def prepare_data(
        self, 
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Prepare data for training or testing
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series, optional
            Target variable (not used, kept for compatibility)
        is_train : bool
            Whether this is training data (fit scaler) or test data (transform only)
            
        Returns:
        --------
        X_prepared : pd.DataFrame
            Prepared features
        """
        logger.info(f"Preparing {'train' if is_train else 'test'} data - Type: {self.data_type}")
        logger.info(f"Data shape: {X.shape}")
        
        X = X.copy()
        
        # Store feature names on first call
        if self.feature_names is None:
            self.feature_names = X.columns.tolist()
        # Drop non-numeric columns for Logistic Regression on raw data
        
        # Handle data based on type
        if self.data_type == 'raw':
            if self.model_type in['random_forest', 'logistic_regression']:
                X = X.select_dtypes(include=[np.number])
                logger.info(f"Dropping non-numeric columns for Logistic Regression or Rf on raw data")
            # Convert object columns to category for SMOTENC
            else:    
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = X[col].astype('category')
                
                # Store categorical column info for SMOTENC (only once)
                if is_train and self.categorical_columns is None:
                    self.categorical_columns = [col for col in X.columns if X[col].dtype.name == 'category']
                    self.categorical_indices = [
                        i for i, col in enumerate(X.columns) 
                        if X[col].dtype.name == 'category'
                    ]
                    logger.info(f"Number of categorical features: {len(self.categorical_indices)}")
                
        else:  # preprocessed
            # All features should be numeric after preprocessing
            if self.categorical_columns is None:
                self.categorical_columns = []
                self.categorical_indices = []
                logger.info("Using preprocessed data - all features are numeric")
        
        # Apply scaling for Logistic Regression with preprocessed data
        # if self.model_type == 'logistic_regression' and self.data_type == 'preprocessed':
        #     if is_train:
        #         self.scaler = StandardScaler()
        #         X = pd.DataFrame(
        #             self.scaler.fit_transform(X),
        #             columns=X.columns,
        #             index=X.index
        #         )
        #         logger.info("Fitted StandardScaler for Logistic Regression")
        #     else:
        #         if self.scaler is not None:
        #             X = pd.DataFrame(
        #                 self.scaler.transform(X),
        #                 columns=X.columns,
        #                 index=X.index
        #             )
        #             logger.info("Applied StandardScaler for Logistic Regression")
        
        return X
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data - test_size: {self.test_size}")
        logger.info(f"Original data shape: {X.shape}")
        logger.info(f"Class distribution: \n{y.value_counts()}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def apply_smote(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE for class balancing
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
            
        Returns:
        --------
        X_train_resampled, y_train_resampled
        """
        if not self.use_smote:
            return X_train, y_train
        
        logger.info("Applying SMOTE for class balancing...")
        
        # Choose SMOTE variant based on data type
        if (
                self.data_type == 'raw'
                and hasattr(self, 'categorical_indices')
                and self.categorical_indices is not None
                and len(self.categorical_indices) > 0
            ):
            # Use SMOTENC for data with categorical features
            smote = SMOTENC(
                categorical_features=self.categorical_indices,
                random_state=self.random_state
            )
            logger.info(f"Using SMOTENC with {len(self.categorical_indices)} categorical features")
        else:
            # Use regular SMOTE for fully numeric data
            smote = SMOTE(random_state=self.random_state)
            logger.info("Using SMOTE for numeric features")
        
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        logger.info(f"Class distribution after SMOTE:")
        logger.info(f"\n{pd.Series(y_train_smote).value_counts()}")
        
        return X_train_smote, y_train_smote
    
    def build_model(self, model_params: Optional[Dict] = None):
        """
        Build the ML model
        
        Parameters:
        -----------
        model_params : dict, optional
            Model hyperparameters
        """
        if self.model_type == 'xgboost':
            if model_params is None:
                model_params = {
                    'n_estimators': 300,
                    'max_depth': 5,
                    'learning_rate': 0.05,
                    'subsample': 0.9,
                    'colsample_bytree': 0.9,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'tree_method': 'hist',
                    'random_state': self.random_state
                }
                
                # Enable categorical only for raw data
                if self.data_type == 'raw':
                    model_params['enable_categorical'] = True
            
            self.model = xgb.XGBClassifier(**model_params)
            logger.info(f"Built XGBoost model")
            
        elif self.model_type == 'random_forest':
            if model_params is None:
                model_params = {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            
            self.model = RandomForestClassifier(**model_params)
            logger.info(f"Built Random Forest model")
            
        elif self.model_type == 'logistic_regression':
            if model_params is None:
                model_params = {
                    'max_iter': 1000,
                    'solver': 'saga',
                    'class_weight': 'balanced',
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
            
            self.model = LogisticRegression(**model_params)
            logger.info(f"Built Logistic Regression model")
            
        else:
            raise ValueError(f"Model type {self.model_type} not supported")
        
        logger.info(f"Model params: {model_params}")
        return self.model
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        model_params: Optional[Dict] = None
    ):
        """
        Train the model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        model_params : dict, optional
            Model hyperparameters
        """
        logger.info(f"Starting {self.model_type} training...")
        
        # Build model if not already built
        if self.model is None:
            self.build_model(model_params)
        
        # Train
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
        
        return self.model
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to predict on
            
        Returns:
        --------
        predictions, probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Apply scaling if needed
        if self.scaler is not None:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        dataset_name: str = "Test"
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True labels
        dataset_name : str
            Name of dataset (for logging)
            
        Returns:
        --------
        metrics, predictions, probabilities
        """
        y_pred, y_proba = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba),
            'precision_0': precision_score(y, y_pred, pos_label=0, zero_division=0),
            'recall_0': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'f1_0': f1_score(y, y_pred, pos_label=0, zero_division=0),
            'precision_1': precision_score(y, y_pred, pos_label=1, zero_division=0),
            'recall_1': recall_score(y, y_pred, pos_label=1, zero_division=0),
            'f1_1': f1_score(y, y_pred, pos_label=1, zero_division=0),
        }
        
        return metrics, y_pred, y_proba
    
    def print_evaluation(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ):
        """
        Print comprehensive evaluation results
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
        """
        # Train evaluation
        train_metrics, y_train_pred, y_train_proba = self.evaluate(
            X_train, y_train, "Train"
        )
        
        # Test evaluation
        test_metrics, y_test_pred, y_test_proba = self.evaluate(
            X_test, y_test, "Test"
        )
        
        # Print results
        print("\n" + "="*60)
        print(f"📊 {self.model_type.upper().replace('_', ' ')} MODEL EVALUATION")
        print(f"    Data Type: {self.data_type.upper()}")
        print(f"    SMOTE: {'Enabled' if self.use_smote else 'Disabled'}")
        print("="*60)
        
        # Prediction distribution
        print("\n📢 Prediction Counts (Test Set):")
        pred_counts = pd.Series(y_test_pred).value_counts().sort_index()
        for label, count in pred_counts.items():
            print(f"   Class {label}: {count}")
        
        # Performance metrics
        print("\n" + "-"*60)
        print("📈 PERFORMANCE METRICS")
        print("-"*60)
        
        print(f"\n{'Metric':<20} {'Train':>12} {'Test':>12}")
        print("-"*46)
        print(f"{'Accuracy':<20} {train_metrics['accuracy']:>12.4f} {test_metrics['accuracy']:>12.4f}")
        print(f"{'ROC-AUC':<20} {train_metrics['roc_auc']:>12.4f} {test_metrics['roc_auc']:>12.4f}")
        
        print(f"\n{'Class 0 Metrics':<20}")
        print(f"{'  Precision':<20} {train_metrics['precision_0']:>12.4f} {test_metrics['precision_0']:>12.4f}")
        print(f"{'  Recall':<20} {train_metrics['recall_0']:>12.4f} {test_metrics['recall_0']:>12.4f}")
        print(f"{'  F1-Score':<20} {train_metrics['f1_0']:>12.4f} {test_metrics['f1_0']:>12.4f}")
        
        print(f"\n{'Class 1 Metrics':<20}")
        print(f"{'  Precision':<20} {train_metrics['precision_1']:>12.4f} {test_metrics['precision_1']:>12.4f}")
        print(f"{'  Recall':<20} {train_metrics['recall_1']:>12.4f} {test_metrics['recall_1']:>12.4f}")
        print(f"{'  F1-Score':<20} {train_metrics['f1_1']:>12.4f} {test_metrics['f1_1']:>12.4f}")
        
        # Confusion matrix
        print("\n" + "-"*60)
        print("🔍 CONFUSION MATRIX (Test Set)")
        print("-"*60)
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # Classification report
        print("\n" + "-"*60)
        print("📋 CLASSIFICATION REPORT (Test Set)")
        print("-"*60)
        print(classification_report(y_test, y_test_pred))
        
        return train_metrics, test_metrics
    
    def save_model(self, filepath: str, preprocessing_pipeline=None):
        """
        Save trained model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save model
        preprocessing_pipeline : sklearn.pipeline.Pipeline, optional
            Preprocessing pipeline to save with model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'data_type': self.data_type,
            'feature_names': self.feature_names,
            'categorical_indices': self.categorical_indices,
            'categorical_columns': self.categorical_columns,
            'preprocessing_pipeline': preprocessing_pipeline
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        if preprocessing_pipeline is not None:
            logger.info(f"Preprocessing pipeline saved with model")
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to load model from
            
        Returns:
        --------
        preprocessing_pipeline : sklearn.pipeline.Pipeline or None
            Loaded preprocessing pipeline if it was saved
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler')
        self.model_type = model_data['model_type']
        self.data_type = model_data['data_type']
        self.feature_names = model_data['feature_names']
        self.categorical_indices = model_data.get('categorical_indices')
        self.categorical_columns = model_data.get('categorical_columns')
        
        preprocessing_pipeline = model_data.get('preprocessing_pipeline')
        
        logger.info(f"Model loaded from {filepath}")
        if preprocessing_pipeline is not None:
            logger.info(f"Preprocessing pipeline loaded with model")
        
        return preprocessing_pipeline



def load_preprocessing_pipeline(filepath: str):
    """
    Load preprocessing pipeline from disk
    
    Parameters:
    -----------
    filepath : str
        Path to load pipeline from
        
    Returns:
    --------
    pipeline : sklearn.pipeline.Pipeline
        Loaded preprocessing pipeline
    """
    pipeline = joblib.load(filepath)
    logger.info(f"Preprocessing pipeline loaded from {filepath}")
    return pipeline


# =============================================================================
# EXPERIMENT RUNNER - Test Multiple Models on Different Data Types
# =============================================================================

# =============================================================================
# EXPERIMENT RUNNER - Test Multiple Models on Different Data Types
# =============================================================================

def run_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    models: list = None,
    data_types: list = None,
    use_smote: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    pipeline_path: str = "models/preprocessing_pipeline.pkl"
):
    """
    Run experiments with multiple models and data types
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features (raw data before preprocessing)
    y : pd.Series
        Target variable
    models : list
        List of model types to test
    data_types : list
        List of data types ('raw', 'preprocessed')
    use_smote : bool
        Whether to use SMOTE
    test_size : float
        Test set size
    random_state : int
        Random state
    pipeline_path : str
        Path to load preprocessing pipeline from
        
    Returns:
    --------
    results : dict
        Dictionary of all results
    """
    if models is None:
        models = ['xgboost', 'random_forest', 'logistic_regression']
    
    if data_types is None:
        data_types = ['preprocessed']  # Default to preprocessed
    
    # Load preprocessing pipeline if needed
    preprocessing_pipeline = None
    if 'preprocessed' in data_types:
        try:
            preprocessing_pipeline = load_preprocessing_pipeline(pipeline_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Preprocessing pipeline not found at '{pipeline_path}'. "
                f"Please ensure the pipeline is saved at this location."
            )
    
    results = {}
    
    for data_type in data_types:
        for model_type in models:
            print(f"\n{'='*80}")
            print(f"Running: {model_type.upper()} on {data_type.upper()} data")
            print(f"{'='*80}")
            
            try:
                # Initialize trainer
                trainer = MLModelTrainer(
                    model_type=model_type,
                    use_smote=use_smote,
                    test_size=test_size,
                    random_state=random_state,
                    data_type=data_type
                )
                
                # Split data FIRST
                X_train, X_test, y_train, y_test = trainer.split_data(X, y)
                
                # Apply preprocessing if needed
                if data_type == 'preprocessed':
                    logger.info("Applying preprocessing pipeline to train data...")
                    X_train = preprocessing_pipeline.fit_transform(X_train)
                    
                    # Handle row dropping if pipeline has it
                    if 'drop_rows' in preprocessing_pipeline.named_steps:
                        row_dropper = preprocessing_pipeline.named_steps['drop_rows']
                        y_train = row_dropper.align_y(y_train)
                        logger.info(f"Aligned y_train after row dropping: {len(y_train)} rows")
                    
                    logger.info("Applying preprocessing pipeline to test data...")
                    X_test = preprocessing_pipeline.transform(X_test)
                    
                    # Convert to DataFrame if needed
                    if isinstance(X_train, np.ndarray):
                        X_train = pd.DataFrame(
                            X_train,
                            columns=[f'feature_{i}' for i in range(X_train.shape[1])]
                        )
                    if isinstance(X_test, np.ndarray):
                        X_test = pd.DataFrame(
                            X_test,
                            columns=[f'feature_{i}' for i in range(X_test.shape[1])]
                        )
                
                # Prepare data (handles categorical conversion and scaling)
                X_train = trainer.prepare_data(X_train, y_train, is_train=True)
                X_test = trainer.prepare_data(X_test, y_test, is_train=False)
                
                # Apply SMOTE only if enabled for this run
                if use_smote:
                    logger.info("✅ Applying SMOTE to training data...")
                    X_train_final, y_train_final = trainer.apply_smote(X_train, y_train)
                else:
                    logger.info("⏭️  Skipping SMOTE...")
                    X_train_final, y_train_final = X_train, y_train
                
                
                # Train
                trainer.train(X_train_final, y_train_final)
                
                # Evaluate
                train_metrics, test_metrics = trainer.print_evaluation(
                    X_train, y_train,
                    X_test, y_test
                )

                # Save model
                model_path = f"models/{model_type}_{data_type}_model.pkl"
                trainer.save_model(model_path, preprocessing_pipeline=preprocessing_pipeline if data_type == 'preprocessed' else None)
                
                                # -----------------------------
                # VISUALIZATION SECTION
                # -----------------------------
                model = trainer.model

                # Predictions
                y_test_pred, y_test_proba = trainer.predict(X_test)
                y_train_pred, y_train_proba = trainer.predict(X_train)

                test_auc = roc_auc_score(y_test, y_test_proba)
                train_auc = roc_auc_score(y_train, y_train_proba)

                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                train_precision = precision_score(y_train, y_train_pred)
                test_precision = precision_score(y_test, y_test_pred)
                train_recall = recall_score(y_train, y_train_pred)
                test_recall = recall_score(y_test, y_test_pred)


                # ===============================================================
                # CASE 1 — LOGISTIC REGRESSION VISUALIZATION
                # ===============================================================
                if model_type == "logistic_regression":

                    # Extract coefficients
                    coefficients = model.coef_[0]
                    feature_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'coefficient': coefficients,
                        'abs_coefficient': np.abs(coefficients)
                    }).sort_values('abs_coefficient', ascending=False)

                    print("\n🎯 Top 15 Important Features (by abs(coef)):")
                    print(feature_importance.head(15)[['feature', 'coefficient']].to_string(index=False))

                    # ---------------------------------------
                    # PLOT
                    # ---------------------------------------
                    fig = plt.figure(figsize=(16, 12))
                    plt.suptitle(f"Logistic Regression With {data_type.title()} Performance")

                    # 1 — Top Coefficient Plot
                    ax1 = plt.subplot(2, 3, 1)
                    top = feature_importance.head(15)
                    colors = ['#DD1C1A' if c > 0 else '#06AED5' for c in top['coefficient']]
                    sns.barplot(data=top, y='feature', x='coefficient', palette=colors, ax=ax1)
                    ax1.axvline(0, color='black')
                    ax1.set_title("Top 15 Feature Coefficients")

                    # 2 — Confusion Matrix
                    ax2 = plt.subplot(2, 3, 2)
                    cm = confusion_matrix(y_test, y_test_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax2)
                    ax2.set_title("Confusion Matrix (Test)")

                    # 3 — ROC Curve
                    ax3 = plt.subplot(2, 3, 3)
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                    ax3.plot(fpr, tpr, label=f"AUC = {test_auc:.4f}", color="#2E86AB")
                    ax3.plot([0,1],[0,1],"k--")
                    ax3.set_title("ROC Curve")
                    ax3.legend()

                    # 4 — Precision Recall Curve
                    ax4 = plt.subplot(2, 3, 4)
                    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
                    ax4.plot(recall, precision, color="#A23B72")
                    ax4.set_title("Precision-Recall Curve")

                    # 5 — Probability Distribution
                    ax5 = plt.subplot(2, 3, 5)
                    ax5.hist(y_test_proba[y_test == 0], bins=40, alpha=0.7, label="Class 0")
                    ax5.hist(y_test_proba[y_test == 1], bins=40, alpha=0.7, label="Class 1")
                    ax5.legend()
                    ax5.set_title("Prediction Probability Distribution")

                    # 6 — Metrics Comparison
                    ax6 = plt.subplot(2, 3, 6)
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy','ROC-AUC','F1','Precision','Recall'],
                        'Train': [train_acc, train_auc, train_f1, train_precision, train_recall],
                        'Test':  [test_acc, test_auc, test_f1, test_precision, test_recall]
                    })
                    x = np.arange(len(metrics_df))
                    w = 0.35
                    ax6.bar(x-w/2, metrics_df['Train'], w, label="Train", color="#06AED5")
                    ax6.bar(x+w/2, metrics_df['Test'], w, label="Test", color="#DD1C1A")
                    ax6.set_xticks(x)
                    ax6.set_xticklabels(metrics_df['Metric'], rotation=45)
                    ax6.legend()
                    ax6.set_ylim([0, 1.05])
                    ax6.set_title("Train vs Test Performance")

                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    plot_dir = Path("plots")
                    plot_dir.mkdir(exist_ok=True)
                    plot_path = plot_dir / f"{model_type}_{data_type}_performance.png"

                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    results[key]["plot_path"] = str(plot_path)
                    print(f"📁 Logistic Regression Plot saved → {plot_path}")


                # ===============================================================
                # CASE 2 — TREE MODELS (RandomForest, XGBoost)
                # ===============================================================
                else:
                    # Use your earlier feature importance visualization
                    # ---------------------------------------
                    # Feature Importances
                    # ---------------------------------------
                    if hasattr(model, "feature_importances_"):
                        feature_importance = pd.DataFrame({
                            "feature": X_train.columns,
                            "importance": model.feature_importances_
                        }).sort_values("importance", ascending=False)
                    else:
                        feature_importance = None

                    fig = plt.figure(figsize=(16, 12))
                    plt.grid(False)
                    plt.suptitle(f"{model_type.replace('_',' ').title()} With {data_type.title()} Performance",
                                 fontsize=18, fontweight='bold', y=0.98)
                    # 1. Feature Importance Plot
                    ax1 = plt.subplot(2, 3, 1)
                    top_features = feature_importance.head(15)
                    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis', ax=ax1)
                    ax1.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
                    ax1.set_xlabel('Importance Score', fontsize=11)
                    ax1.set_ylabel('Features', fontsize=11)

                    # 2. Confusion Matrix
                    ax2 = plt.subplot(2, 3, 2)
                    cm = confusion_matrix(y_test, y_test_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                                annot_kws={'size': 14, 'weight': 'bold'})
                    ax2.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
                    ax2.set_ylabel('Actual', fontsize=11)
                    ax2.set_xlabel('Predicted', fontsize=11)

                    # 3. ROC Curve
                    ax3 = plt.subplot(2, 3, 3)
                    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
                    ax3.plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.4f})', linewidth=2.5, color='#2E86AB')
                    ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1.5, alpha=0.7)
                    ax3.set_xlabel('False Positive Rate', fontsize=11)
                    ax3.set_ylabel('True Positive Rate', fontsize=11)
                    ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
                    ax3.legend(loc='lower right', fontsize=10)

                    # 4. Precision-Recall Curve
                    ax4 = plt.subplot(2, 3, 4)
                    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
                    ax4.plot(recall, precision, linewidth=2.5, color='#A23B72')
                    ax4.set_xlabel('Recall', fontsize=11)
                    ax4.set_ylabel('Precision', fontsize=11)
                    ax4.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')

                    # 5. Prediction Probability Distribution
                    ax5 = plt.subplot(2, 3, 5)
                    ax5.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='Class 0 (No Default)', 
                            color='#06AED5', edgecolor='black', linewidth=0.5)
                    ax5.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Class 1 (Default)', 
                            color='#DD1C1A', edgecolor='black', linewidth=0.5)
                    ax5.set_xlabel('Predicted Probability', fontsize=11)
                    ax5.set_ylabel('Frequency', fontsize=11)
                    ax5.set_title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
                    ax5.legend(fontsize=10)

                    # 6. Performance Metrics Comparison
                    ax6 = plt.subplot(2, 3, 6)
                    metrics_df = pd.DataFrame({
                        'Metric': ['Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall'],
                        'Train': [train_acc, train_auc, train_f1, train_precision, train_recall],
                        'Test': [test_acc, test_auc, test_f1, test_precision, test_recall]
                    })
                    x = np.arange(len(metrics_df))
                    width = 0.35
                    bars1 = ax6.bar(x - width/2, metrics_df['Train'], width, label='Train', color='#06AED5', alpha=0.8)
                    bars2 = ax6.bar(x + width/2, metrics_df['Test'], width, label='Test', color='#DD1C1A', alpha=0.8)
                    ax6.set_xlabel('Metrics', fontsize=11)
                    ax6.set_ylabel('Score', fontsize=11)
                    ax6.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
                    ax6.set_xticks(x)
                    ax6.set_xticklabels(metrics_df['Metric'], rotation=45, ha='right')
                    ax6.legend(fontsize=10)
                    ax6.set_ylim([0, 1.05])

                    # Add value labels on bars
                    for bars in [bars1, bars2]:
                        for bar in bars:
                            height = bar.get_height()
                            ax6.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

                    plt.tight_layout()
                    plot_dir = Path("plots")
                    plot_dir.mkdir(exist_ok=True)
                    plot_path = plot_dir / f"{model_type}_{data_type}_performance.png"

                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

        
                    print(f"📁 Tree Model Plot saved → {plot_path}")


                # Store results
                key = f"{model_type}_{data_type}"
                results[key] = {
                    'trainer': trainer,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'model_path': model_path,
                    'pipeline_path': pipeline_path if data_type == 'preprocessed' else None
                }
                
                logger.info(f"✅ Completed: {model_type} on {data_type} data")
                
            except Exception as e:
                logger.error(f"❌ Failed: {model_type} on {data_type} data - {str(e)}")
                import traceback
                traceback.print_exc()
                results[f"{model_type}_{data_type}"] = {'error': str(e)}
    
    return results


# =============================================================================
# USAGE EXAMPLES
# =============================================================================



def example_with_raw_data_dropna():
    """
    Example: Train models on RAW data
    """
    # Load raw data
    df = pd.read_csv('raw_data/train.csv')
    
    # Drop NA
    df = df.dropna()
    
    # Separate X and y
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Run experiments with raw data
    results = run_experiments(
        X, y,
        models=['random_forest', 'logistic_regression'],
        data_types=['raw'],
        use_smote=False
    )
    
    return results

def xgboost_example_with_raw_data():
    """
    Example: Train XGBoost model on RAW data
    """
    # Load raw data
    df = pd.read_csv('raw_data/train.csv')
    
    
    # Separate X and y
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Run experiments with raw data
    results = run_experiments(
        X, y,
        models=['xgboost'],
        data_types=['raw'],
        use_smote= False
    )
    
    return results

def example_with_preprocessed_data(pipeline_path: str = 'models/preprocessing_pipeline.pkl'):
    """
    Example: Train models on PREPROCESSED data
    Loads the preprocessing pipeline from disk automatically
    
    Parameters:
    -----------
    pipeline_path : str
        Path to saved preprocessing pipeline
    """
    # Load raw data
    df = pd.read_csv('raw_data/train.csv')
    
    # Separate X and y (don't drop NA - let pipeline handle it)
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Run experiments - pipeline will be loaded automatically
    results = run_experiments(
        X, y,
        models=['xgboost', 'random_forest', 'logistic_regression'],
        data_types=['preprocessed'],
        use_smote=True,
        pipeline_path=pipeline_path
    )
    
    return results



def example_single_model_preprocessed(pipeline_path: str = 'models/preprocessing_pipeline.pkl'):
    """
    Example: Train a single model on preprocessed data
    Loads preprocessing pipeline from disk
    
    Parameters:
    -----------
    pipeline_path : str
        Path to saved preprocessing pipeline
    """
    # Load raw data
    df = pd.read_csv('raw_data/train.csv')
    
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Load preprocessing pipeline
    preprocessing_pipeline = load_preprocessing_pipeline(pipeline_path)
    
    # Initialize trainer
    trainer = MLModelTrainer(
        model_type='logistic_regression',
        use_smote=True,
        data_type='preprocessed'
    )
    
    # STEP 1: Split data FIRST
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # STEP 2: Apply preprocessing separately to train and test
    logger.info("Preprocessing training data...")
    X_train_prep = preprocessing_pipeline.fit_transform(X_train)
    
    # Handle row dropping if your pipeline has it
    if 'drop_rows' in preprocessing_pipeline.named_steps:
        row_dropper = preprocessing_pipeline.named_steps['drop_rows']
        y_train = row_dropper.align_y(y_train)
    
    logger.info("Preprocessing test data...")
    X_test_prep = preprocessing_pipeline.transform(X_test)
    
    # Convert to DataFrame if needed
    if isinstance(X_train_prep, np.ndarray):
        X_train_prep = pd.DataFrame(X_train_prep, columns=[f'f_{i}' for i in range(X_train_prep.shape[1])])
        X_test_prep = pd.DataFrame(X_test_prep, columns=[f'f_{i}' for i in range(X_test_prep.shape[1])])
    
    # STEP 3: Prepare data (handles scaling for logistic regression)
    X_train_prep = trainer.prepare_data(X_train_prep, y_train, is_train=True)
    X_test_prep = trainer.prepare_data(X_test_prep, y_test, is_train=False)
    
    # STEP 4: Apply SMOTE
    X_train_smote, y_train_smote = trainer.apply_smote(X_train_prep, y_train)
    
    # STEP 5: Train
    trainer.train(X_train_smote, y_train_smote)
    
    # STEP 6: Evaluate
    train_metrics, test_metrics = trainer.print_evaluation(
        X_train_prep, y_train,
        X_test_prep, y_test
    )
    
    # STEP 7: Save model with pipeline
    trainer.save_model('models/my_preprocessed_model.pkl', preprocessing_pipeline=preprocessing_pipeline)
    
    return trainer


def example_load_and_predict(
    model_path: str = 'models/logistic_regression_preprocessed_model.pkl',
    data_path: str = 'raw_data/test.csv'
):
    """
    Example: Load saved model and make predictions on new data
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
    data_path : str
        Path to new data for predictions
    """
    # Load new data
    df_new = pd.read_csv(data_path)
    
    # Load model
    trainer = MLModelTrainer()
    preprocessing_pipeline = trainer.load_model(model_path)
    
    logger.info(f"Loaded {trainer.model_type} model (data_type: {trainer.data_type})")
    
    # Prepare new data
    if trainer.data_type == 'preprocessed' and preprocessing_pipeline is not None:
        # Apply preprocessing
        logger.info("Applying preprocessing to new data...")
        X_new_prep = preprocessing_pipeline.transform(df_new)
        
        # Convert to DataFrame if needed
        if isinstance(X_new_prep, np.ndarray):
            X_new_prep = pd.DataFrame(
                X_new_prep,
                columns=[f'f_{i}' for i in range(X_new_prep.shape[1])]
            )
        
        # Prepare (applies scaling if needed)
        X_new_prep = trainer.prepare_data(X_new_prep, is_train=False)
        
        # Make predictions
        predictions, probabilities = trainer.predict(X_new_prep)
        
    else:
        # Raw data - just prepare
        X_new = trainer.prepare_data(df_new, is_train=False)
        predictions, probabilities = trainer.predict(X_new)
    
    # Create results dataframe
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    logger.info(f"Made predictions on {len(results)} samples")
    logger.info(f"Prediction distribution:\n{results['prediction'].value_counts()}")
    
    return results


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ML TRAINING PIPELINE")
    print("="*80)
    
# Train on raw data instead
    print("\nTraining on RAW data drop na as fallback...")
    results_raw = example_with_raw_data_dropna()
    print('='*80)
    print("Training on RAW data not drop na.")
    results_raw_no_dropna = xgboost_example_with_raw_data()
    
    print('\nCompleted examples.')
    # Check if preprocessing pipeline exists
    import os
    pipeline_path = 'models/preprocessing_pipeline.pkl'
    
    if os.path.exists(pipeline_path):
        print(f"\n✅ Found preprocessing pipeline at: {pipeline_path}")
        
        # EXAMPLE 1: Train all models on preprocessed data
        print("\n" + "="*80)
        print("Training models on PREPROCESSED data...")
        print("="*80)
        results_preprocessed = example_with_preprocessed_data(pipeline_path)
        
        # Print summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        for key, result in results_preprocessed.items():
            if 'error' in result:
                print(f"❌ {key}: FAILED - {result['error']}")
            else:
                test_metrics = result['test_metrics']
                print(f"✅ {key}:")
                print(f"   - Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"   - Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
                print(f"   - Model saved:   {result['model_path']}")
        
        # EXAMPLE 2: Make predictions with saved model
        """
        print("\n" + "="*80)
        print("Making predictions on new data...")
        print("="*80)
        predictions = example_load_and_predict(
            model_path='models/xgboost_preprocessed_model.pkl',
            data_path='raw_data/test.csv'
        )
        print(predictions.head())
        """
        
    else:
        print(f"\n⚠️  Preprocessing pipeline not found at: {pipeline_path}")
        print("="*80)
    