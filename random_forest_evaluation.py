import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gc
import os
import logging
from typing import Optional, Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set up logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intrusion_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DecisionTree:
    """Production-ready Decision Tree implementation from scratch for multiclass classification"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2, 
                 min_samples_leaf: int = 1, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Calculate Gini impurity for multiclass labels"""
        if len(y) == 0:
            return 0.0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return 1.0 - sum(p**2 for p in probabilities)
    
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find the best split point for a node"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        # Limit number of features to check for faster training
        max_features_to_check = min(10, n_features)
        features_to_check = np.random.choice(n_features, size=max_features_to_check, replace=False)
        
        for feature in features_to_check:
            # Sample thresholds for faster training
            feature_values = X[:, feature]
            unique_values = np.unique(feature_values)
            
            # Limit number of thresholds to check
            if len(unique_values) > 20:
                thresholds = np.percentile(feature_values, np.linspace(10, 90, 20))
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calculate weighted Gini impurity
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                left_weight = np.sum(left_mask) / len(y)
                right_weight = np.sum(right_mask) / len(y)
                
                weighted_gini = left_weight * left_gini + right_weight * right_gini
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _create_leaf(self, y: np.ndarray) -> Dict[str, Any]:
        """Create a leaf node with majority class"""
        counts = Counter(y)
        most_common = max(counts.items(), key=lambda x: x[1])
        return {'type': 'leaf', 'prediction': most_common[0]}
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Dict[str, Any]:
        """Recursively build the decision tree"""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return self._create_leaf(y)
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf(y)
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Check if split is valid
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return self._create_leaf(y)
        
        # Create node
        node = {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }
        
        return node
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """Train the decision tree"""
        logger.info(f"Training decision tree with {len(X)} samples and {X.shape[1]} features")
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x: np.ndarray, node: Dict[str, Any]) -> int:
        """Predict a single sample"""
        if node['type'] == 'leaf':
            return node['prediction']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict for multiple samples"""
        if self.tree is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        return np.array([self._predict_single(x, self.tree) for x in X])

class RandomForest:
    """Production-ready Random Forest implementation from scratch for multiclass classification"""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 10, 
                 min_samples_split: int = 2, min_samples_leaf: int = 1, 
                 max_features: str = 'sqrt', bootstrap: bool = True, 
                 random_state: int = 42, n_jobs: int = -1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.trees = []
        self.feature_importances_ = None
        
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create bootstrap sample with replacement"""
        n_samples = len(y)
        indices = np.random.choice(n_samples, size=int(n_samples), replace=True)
        return X[indices], y[indices]
    
    def _get_feature_subset(self, n_features: int) -> np.ndarray:
        """Get random subset of features"""
        if isinstance(self.max_features, str):
            if self.max_features == 'sqrt':
                n_subset = int(np.sqrt(n_features))
            elif self.max_features == 'log2':
                n_subset = int(np.log2(n_features))
            else:
                n_subset = n_features
        else:
            n_subset = min(self.max_features, n_features)
        
        return np.random.choice(n_features, size=int(n_subset), replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """Train the Random Forest"""
        logger.info(f"Training Random Forest with {self.n_estimators} trees")
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            if i % 10 == 0:
                logger.info(f"Training tree {i+1}/{self.n_estimators}")
            
            # Create bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Get feature subset
            feature_subset = self._get_feature_subset(n_features)
            X_subset = X_sample[:, feature_subset]
            
            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            tree.fit(X_subset, y_sample)
            
            # Store tree with feature mapping
            self.trees.append((tree, feature_subset))
            
            # Update feature importances
            for feature_idx in feature_subset:
                self.feature_importances_[feature_idx] += 1
        
        # Normalize feature importances
        self.feature_importances_ /= self.n_estimators
        
        logger.info("Random Forest training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority voting"""
        if not self.trees:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        predictions = []
        
        for tree, feature_subset in self.trees:
            X_subset = X[:, feature_subset]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Majority vote
        predictions = np.array(predictions)
        final_predictions = []
        
        for i in range(len(X)):
            votes = predictions[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            final_predictions.append(unique[np.argmax(counts)])
        
        return np.array(final_predictions)

class MulticlassIntrusionDetector:
    """Production-ready Multiclass Network Intrusion Detection System based on research paper"""
    
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.training_time = None
        self.model_size = None
        self.class_mapping = None
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess CICIDS-2017 data with research paper methodology"""
        logger.info("Starting data loading and preprocessing using research paper methodology")
        
        files = [
            'cicids-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
            'cicids-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv',
            'cicids-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
            'cicids-2017/Monday-WorkingHours.pcap_ISCX.csv',
            'cicids-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv',
            'cicids-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
            'cicids-2017/Tuesday-WorkingHours.pcap_ISCX.csv',
            'cicids-2017/Wednesday-workingHours.pcap_ISCX.csv'
        ]
        
        all_data = []
        total_processed = 0
        
        for file in files:
            if not os.path.exists(file):
                logger.warning(f"File not found: {file}")
                continue
                
            logger.info(f"Processing {file}")
            
            try:
                # Process file in chunks for memory efficiency
                for chunk in pd.read_csv(file, chunksize=100000):
                    # Clean column names
                    chunk.columns = [col.strip() for col in chunk.columns]
                    
                    # Find label column
                    label_col = None
                    for col in chunk.columns:
                        if col.lower() == 'label':
                            label_col = col
                            break
                    
                    if label_col is None:
                        logger.warning(f"No label column found in {file}")
                        continue
                    
                    # Use ALL features as per research paper
                    all_features = [label_col] + [col for col in chunk.columns if col != label_col]
                    
                    try:
                        chunk = chunk[all_features]
                    except KeyError as e:
                        logger.warning(f"Column selection error in {file}: {e}")
                        continue
                    
                    all_data.append(chunk)
                    total_processed += len(chunk)
                    
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data could be processed from any file")
        
        data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Initial dataset shape: {data.shape}")
        logger.info(f"Total samples processed: {total_processed}")
        
        return data
    
    def apply_research_paper_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the research paper's sophisticated preprocessing methodology"""
        logger.info("Applying research paper preprocessing methodology")
        
        # Step 1: Handle missing values and infinite values
        logger.info("Step 1: Handling missing and infinite values")
        
        # Handle infinite values in Flow Bytes/s and Flow Packets/s
        if 'Flow Bytes/s' in data.columns:
            # Create binary features for infinite values
            data['Has Infinite Flow Bytes'] = np.where(np.isinf(data['Flow Bytes/s']), 1, 0)
            data['Has Infinite Flow Packets'] = np.where(np.isinf(data['Flow Packets/s']), 1, 0)
            
            # Replace infinite values with median
            data['Flow Bytes/s'] = data['Flow Bytes/s'].replace([np.inf, -np.inf], np.nan)
            data['Flow Packets/s'] = data['Flow Packets/s'].replace([np.inf, -np.inf], np.nan)
            
            # Impute with median
            data['Flow Bytes/s'] = data['Flow Bytes/s'].fillna(data['Flow Bytes/s'].median())
            data['Flow Packets/s'] = data['Flow Packets/s'].fillna(data['Flow Packets/s'].median())
        
        # Step 2: Drop constant columns (as per research paper)
        logger.info("Step 2: Dropping constant columns")
        constant_columns = [
            'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 
            'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 
            'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
        ]
        
        for col in constant_columns:
            if col in data.columns:
                data = data.drop(col, axis=1)
                logger.info(f"Dropped constant column: {col}")
        
        # Step 3: Drop highly correlated features (as per research paper)
        logger.info("Step 3: Dropping highly correlated features")
        correlated_columns = [
            'Total Backward Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Std',
            'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Fwd IAT Total', 'Fwd IAT Max',
            'Fwd Packets/s', 'Packet Length Std', 'SYN Flag Count', 'CWE Flag Count',
            'ECE Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size',
            'Fwd Header Length.1', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
            'Subflow Bwd Bytes', 'Idle Mean', 'Idle Max', 'Idle Min'
        ]
        
        for col in correlated_columns:
            if col in data.columns:
                data = data.drop(col, axis=1)
                logger.info(f"Dropped correlated column: {col}")
        
        # Step 4: Multiclass labeling (as per research paper)
        logger.info("Step 4: Applying multiclass labeling")
        label_col = None
        for col in data.columns:
            if col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Label column not found")
        
        # Create multiclass labels
        def create_multiclass_label(label):
            if label == 'BENIGN':
                return 0  # BENIGN
            elif 'DDoS' in label:
                return 1  # DDoS
            elif any(attack in label for attack in ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest']):
                return 2  # DoS
            elif 'PortScan' in label:
                return 3  # PortScan
            elif 'Bot' in label:
                return 4  # Bot
            elif any(attack in label for attack in ['Web Attack - Brute Force', 'Web Attack - XSS', 'Web Attack - Sql Injection']):
                return 5  # WebAttack
            elif any(attack in label for attack in ['FTP-Patator', 'SSH-Patator']):
                return 6  # Patator
            else:
                return 0  # Default to BENIGN
        
        data['multiclass_label'] = data[label_col].apply(create_multiclass_label)
        
        # Create class mapping
        self.class_mapping = {
            0: 'BENIGN',
            1: 'DDoS', 
            2: 'DoS',
            3: 'PortScan',
            4: 'Bot',
            5: 'WebAttack',
            6: 'Patator'
        }
        
        # Step 5: Data sampling (as per research paper)
        logger.info("Step 5: Applying data sampling")
        
        # Take 25% of BENIGN class to balance
        benign_data = data[data['multiclass_label'] == 0]
        if len(benign_data) > 0:
            benign_sample = benign_data.sample(frac=0.25, random_state=42)
            other_data = data[data['multiclass_label'] != 0]
            data = pd.concat([benign_sample, other_data], ignore_index=True)
        
        # Remove rare classes (Infiltration, Heartbleed) as per research paper
        data = data[data['multiclass_label'].isin([0, 1, 2, 3, 4, 5, 6])]
        
        # Drop original label column
        data = data.drop(label_col, axis=1)
        
        logger.info(f"Final dataset shape after preprocessing: {data.shape}")
        logger.info("Class distribution:")
        for class_id, class_name in self.class_mapping.items():
            count = len(data[data['multiclass_label'] == class_id])
            logger.info(f"  {class_name}: {count}")
        
        return data
    
    def train_model(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Train Random Forest model with production-ready methodology"""
        logger.info("Starting model training")
        start_time = datetime.now()
        
        # Separate features and target
        X = data.drop('multiclass_label', axis=1)
        y = data['multiclass_label']
        
        # Split data FIRST (before any scaling) - prevents data leakage
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Basic scaling AFTER splitting (no data leakage)
        logger.info("Scaling features (training data only)")
        X_train_scaled = (X_train - X_train.min()) / (X_train.max() - X_train.min())
        X_test_scaled = (X_test - X_train.min()) / (X_train.max() - X_train.min())
        
        # Handle any remaining NaN values
        X_train_scaled = X_train_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)
        
        self.feature_names = X.columns.tolist()
        
        # Train Random Forest from scratch
        logger.info("Training Random Forest from scratch")
        self.model = RandomForest(
            n_estimators=50,  # Full production model
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        self.model.fit(X_train_scaled.values, y_train.values)
        
        self.training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance with production-ready metrics"""
        logger.info("Evaluating model performance")
        
        if self.model is None:
            logger.error("Model not trained yet")
            return {}
        
        y_pred = self.model.predict(X_test.values)
        
        # Calculate metrics manually
        accuracy = np.mean(y_pred == y_test.values)
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_id, class_name in self.class_mapping.items():
            # Binary classification for each class
            y_binary = (y_test.values == class_id).astype(int)
            pred_binary = (y_pred == class_id).astype(int)
            
            tp = np.sum((pred_binary == 1) & (y_binary == 1))
            fp = np.sum((pred_binary == 1) & (y_binary == 0))
            fn = np.sum((pred_binary == 0) & (y_binary == 1))
            tn = np.sum((pred_binary == 0) & (y_binary == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': np.sum(y_binary)
            }
        
        logger.info(f"Model Performance:")
        logger.info(f"  Overall Accuracy: {accuracy:.4f}")
        
        for class_name, metrics in class_metrics.items():
            logger.info(f"  {class_name}:")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"    Support: {metrics['support']}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test.values, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Baseline comparison
        from sklearn.dummy import DummyClassifier
        dummy = DummyClassifier(strategy='most_frequent', random_state=42)
        dummy.fit(X_test, y_test)
        dummy_pred = dummy.predict(X_test)
        dummy_acc = np.mean(dummy_pred == y_test)
        
        logger.info(f"Baseline (most frequent): {dummy_acc:.4f}")
        logger.info(f"Improvement: {accuracy - dummy_acc:.4f}")
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm.tolist()
        }
    
    def save_model(self, filepath: str = 'models/intrusion_detector_rf.joblib') -> None:
        """Save the trained model with metadata"""
        logger.info(f"Saving model to {filepath}")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model is not None:
            import joblib
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'class_mapping': self.class_mapping,
                'training_time': self.training_time,
                'model_size': len(self.model.trees),
                'timestamp': datetime.now().isoformat()
            }
            joblib.dump(model_data, filepath, compress=3)
            logger.info(f"Model saved successfully to {filepath}")
        else:
            logger.error("No model to save. Please train a model first.")
    
    def save_metrics(self, metrics: Dict[str, Any], 
                    timestamp: Optional[str] = None) -> None:
        """Save model metrics with production metadata"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        os.makedirs('metrics', exist_ok=True)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame([{
            'accuracy': metrics['accuracy'],
            'timestamp': timestamp,
            'model': 'Random Forest (from scratch)',
            'implementation': 'Custom implementation',
            'training_time': self.training_time,
            'model_size': len(self.model.trees) if self.model else 0,
            'features_used': len(self.feature_names) if self.feature_names else 0,
            'classes': len(self.class_mapping) if self.class_mapping else 0
        }])
        
        metrics_file = f'metrics/model_metrics_{timestamp}.csv'
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save per-class metrics
        class_metrics_df = pd.DataFrame()
        for class_name, class_metrics in metrics['class_metrics'].items():
            class_metrics_df = pd.concat([class_metrics_df, pd.DataFrame([{
                'class': class_name,
                'precision': class_metrics['precision'],
                'recall': class_metrics['recall'],
                'f1_score': class_metrics['f1_score'],
                'support': class_metrics['support'],
                'timestamp': timestamp
            }])], ignore_index=True)
        
        class_metrics_file = f'metrics/class_metrics_{timestamp}.csv'
        class_metrics_df.to_csv(class_metrics_file, index=False)
        logger.info(f"Class metrics saved to {class_metrics_file}")

def main():
    """Main execution function with production-ready error handling"""
    logger.info("=" * 80)
    logger.info("Multiclass Network Intrusion Detection with Random Forest (From Scratch)")
    logger.info("Production-ready implementation based on research paper methodology")
    logger.info("=" * 80)
    
    detector = MulticlassIntrusionDetector()
    
    try:
        # Load and preprocess data
        logger.info("STEP 1: Loading and preprocessing data...")
        data = detector.load_and_preprocess_data()
        
        # Apply research paper preprocessing
        logger.info("STEP 2: Applying research paper preprocessing...")
        data = detector.apply_research_paper_preprocessing(data)
        
        # Train model
        logger.info("STEP 3: Training Random Forest model...")
        X_train, X_test, y_train, y_test = detector.train_model(data)
        
        # Evaluate model
        logger.info("STEP 4: Evaluating model...")
        metrics = detector.evaluate_model(X_test, y_test)
        
        # Save results
        logger.info("STEP 5: Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        detector.save_metrics(metrics, timestamp)
        detector.save_model(f'models/intrusion_detector_rf_{timestamp}.joblib')
        
        logger.info("Training complete!")
        logger.info(f"Results saved with timestamp: {timestamp}")
        
    except Exception as e:
        logger.error(f"Critical error during execution: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 