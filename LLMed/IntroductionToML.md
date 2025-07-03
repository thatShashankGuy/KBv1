# Machine Learning

## Definition
A branch of artificial intelligence focused on statistical pattern recognition and automatic inference of predictive models from data.

## Workflow
1. **Data collection**: Assemble a dataset and partition into training, validation and test sets.  
2. **Training**: Apply a learning algorithm to the training set to adjust model parameters.  
3. **Model**: The trained artifact (with fixed parameters) ready for inference.  
4. **Evaluation**: Run the model on unseen data (validation/test set) and compute performance metrics (e.g., accuracy, precision, recall, RMSE).  
5. **Optimization loop**:  
   - Compute a loss function (difference between predicted and true labels).  
   - Propagate gradients (e.g., via backpropagation in neural networks) to update parameters.  
   - Use the validation set for hyperparameter tuning and early stopping to prevent overfitting.  
6. **Deployment & monitoring**:  
   - Serve the model in production.  
   - Monitor for data drift and model degradation.  
   - Retrain periodically on updated data.

## Data & Variables
- **Features** (input variables, X)  
- **Target/Label** (response variable, y) in supervised tasks  
- **Unlabeled data** for unsupervised tasks

> “Garbage in, garbage out”: Model quality critically depends on data quality, feature engineering, and label correctness.

## Learning Paradigms
- **Supervised learning**: Learns from labeled examples (classification, regression).  
- **Unsupervised learning**: Discovers structure in unlabeled data (clustering, dimensionality reduction).  
- **Reinforcement learning**: Learns policies via reward signals through interaction with an environment.

## Task Categories
- **Classification**: Assigns inputs to discrete classes (e.g., spam detection).  
- **Regression**: Predicts continuous values (e.g., price forecasting).  
- **Clustering**: Groups similar instances (e.g., customer segmentation).  
- **Dimensionality Reduction**: Projects data into lower-dimensional space (e.g., PCA, t-SNE).

## Model Families
- **Instance-based**: k-Nearest Neighbors (
