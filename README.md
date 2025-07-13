# Network Intrusion Detection System using Machine Learning

## Overview

This project implements a production-ready Network Intrusion Detection System (NIDS) using custom Random Forest algorithms from scratch. The system is designed to detect various types of network attacks in real-time traffic using the CICIDS-2017 dataset, achieving high accuracy in multiclass classification of network anomalies.

## Project Scope

The primary objective is to develop an anomaly-based intrusion detection system that can identify multiple types of network attacks including DDoS, DoS, PortScan, Bot, WebAttack, and Patator attacks. The system uses machine learning techniques to analyze network traffic patterns and classify them as either benign or malicious.

## Dataset

The project utilizes the CICIDS-2017 dataset from the Canadian Institute for Cybersecurity, which contains:
- 2.8 million network traffic instances
- 79 features representing network flow characteristics
- 7 attack categories plus benign traffic
- Real-world network traffic patterns

## Methodology

### Data Preprocessing
Following the research paper methodology, the preprocessing pipeline includes:
- Feature engineering with 50 carefully selected features
- Handling of infinite values and missing data
- Removal of constant and highly correlated features
- Multiclass labeling for 7 attack types
- Stratified sampling to maintain class distribution

### Model Implementation
The Random Forest algorithm is implemented from scratch without relying on external ML libraries:
- Custom Decision Tree implementation
- Bootstrap sampling for ensemble diversity
- Feature subset selection for each tree
- Majority voting for final predictions
- Gini impurity for split criteria

### Training Process
- Full dataset utilization (1.1M samples after preprocessing)
- 50 trees in the ensemble
- 80:20 train-test split with stratification
- Proper scaling to prevent data leakage
- Comprehensive logging and error handling

## Results

### Performance Metrics
- **Overall Accuracy**: 98.71%
- **Training Time**: 15 hours (custom implementation)
- **Features Used**: 49 engineered features
- **Classes**: 7 attack types + benign

### Per-Class Performance
| Attack Type | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| BENIGN      | 99.08%    | 98.39% | 98.73%   | 113,766 |
| DDoS        | 100.00%   | 99.77% | 99.88%   | 25,606  |
| DoS         | 96.61%    | 99.53% | 98.05%   | 50,532  |
| PortScan    | 99.84%    | 99.89% | 99.87%   | 31,786  |
| Bot         | 85.56%    | 61.83% | 71.79%   | 393     |
| Patator     | 100.00%   | 78.79% | 88.13%   | 2,767   |

### Key Achievements
- Successfully implemented Random Forest from scratch
- Achieved near-research paper accuracy (98.71% vs 99.89%)
- Proper multiclass classification for 7 attack types
- Production-ready code with comprehensive logging
- No data leakage through proper train-test splitting

## Project Structure

```
IDS_anomaly-based/
├── random_forest_evaluation.py    # Main implementation
├── models/                        # Trained models
│   └── intrusion_detector_rf_20250713_102856.joblib
├── metrics/                       # Performance metrics
│   ├── model_metrics_20250713_102856.csv
│   └── class_metrics_20250713_102856.csv
├── visualizations/                # Performance plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── correlation_matrix.png
└── README.md
```

## Installation and Usage

### Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn (for evaluation only)

### Running the Project
```bash
python random_forest_evaluation.py
```

The system will:
1. Load and preprocess the CICIDS-2017 dataset
2. Apply research paper methodology for feature engineering
3. Train custom Random Forest from scratch
4. Evaluate performance with multiclass metrics
5. Save results and trained model

## Future Work

### Real-Time IDS Implementation
The next phase involves developing a real-time intrusion detection system using eBPF (extended Berkeley Packet Filter) technology. This will enable:
- Kernel-level packet filtering and analysis
- User-space processing of network traffic
- Real-time anomaly detection without performance impact
- Integration with existing network infrastructure

### Advanced Anomaly Detection
For detecting zero-day attacks and novel threats, the system will be enhanced with:
- **LSTM (Long Short-Term Memory) Networks**: For temporal pattern analysis in network traffic
- **Autoencoder Neural Networks**: For unsupervised anomaly detection
- **Isolation Forests**: For detecting outliers in high-dimensional data
- **One-Class SVM**: For learning normal behavior patterns

### Dataset Expansion
Future work will incorporate additional datasets:
- **UGR'16 Dataset**: For validation across different network environments
- **UNSW-NB15 Dataset**: For broader attack type coverage
- **Custom Network Traces**: For organization-specific threat modeling

### Production Deployment
The system will be adapted for production environments with:
- Docker containerization for easy deployment
- REST API for integration with security tools
- Real-time alerting and notification systems
- Performance monitoring and logging
- Integration with SIEM (Security Information and Event Management) systems

## Technical Details

### Anomaly-Based Detection
This system implements anomaly-based intrusion detection, which:
- Learns normal network behavior patterns
- Identifies deviations from established baselines
- Detects previously unseen attack types
- Provides early warning for zero-day threats

### Performance Considerations
- Custom implementation allows for optimization specific to network security
- Memory-efficient processing for large-scale deployment
- Scalable architecture for enterprise networks
- Real-time processing capabilities
