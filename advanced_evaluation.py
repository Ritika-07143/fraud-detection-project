"""
Advanced Model Evaluation Metrics
Includes: Calibration, Threshold Optimization, Cost Analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class ThresholdOptimization:
    """
    Find optimal classification threshold
    """
    
    @staticmethod
    def optimize_threshold(y_true, y_pred_proba, metric='f1'):
        """
        Find threshold that maximizes specified metric
        """
        thresholds = np.arange(0, 1.01, 0.01)
        scores = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            if metric == 'f1':
                from sklearn.metrics import f1_score
                score = f1_score(y_true, y_pred)
            elif metric == 'precision':
                from sklearn.metrics import precision_score
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                from sklearn.metrics import recall_score
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == 'balanced':
                # Balance between precision and recall
                from sklearn.metrics import precision_score, recall_score
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                score = 2 * (p * r) / (p + r + 1e-10)
            
            scores.append(score)
        
        optimal_threshold = thresholds[np.argmax(scores)]
        optimal_score = max(scores)
        
        return optimal_threshold, optimal_score, thresholds, scores
    
    @staticmethod
    def cost_sensitive_threshold(y_true, y_pred_proba, 
                                  cost_fp=1, cost_fn=5):
        """
        Optimize threshold based on business costs
        
        Args:
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative
        """
        thresholds = np.arange(0, 1.01, 0.01)
        total_costs = []
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate total cost
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            total_costs.append(total_cost)
        
        optimal_threshold = thresholds[np.argmin(total_costs)]
        min_cost = min(total_costs)
        
        return optimal_threshold, min_cost, thresholds, total_costs


class ModelCalibration:
    """
    Model probability calibration
    """
    
    @staticmethod
    def calibrate_predictions(model, X_train, y_train, X_test, method='isotonic'):
        """
        Calibrate probability predictions
        """
        calibrated_model = CalibratedClassifierCV(
            model,
            method=method,
            cv=5
        )
        
        calibrated_model.fit(X_train, y_train)
        y_pred_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        
        return calibrated_model, y_pred_calibrated
    
    @staticmethod
    def plot_calibration_curve(y_true, y_pred_proba, model_name):
        """
        Plot calibration curve
        """
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba,
            n_bins=10
        )
        
        return prob_true, prob_pred


class AdvancedMetrics:
    """
    Additional evaluation metrics
    """
    
    @staticmethod
    def calculate_business_metrics(y_true, y_pred, avg_fraud_amount=125):
        """
        Calculate business-relevant metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Financial impact
        fraud_prevented = tp * avg_fraud_amount
        false_alarms_cost = fp * 5  # Cost to review
        missed_frauds_cost = fn * avg_fraud_amount
        
        net_benefit = fraud_prevented - false_alarms_cost - missed_frauds_cost
        
        return {
            'fraud_prevented': fraud_prevented,
            'false_alarms_cost': false_alarms_cost,
            'missed_frauds_cost': missed_frauds_cost,
            'net_benefit': net_benefit,
            'prevention_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'accuracy_on_fraud': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
    
    @staticmethod
    def calculate_lift_chart(y_true, y_pred_proba, bins=10):
        """
        Calculate lift at different thresholds
        """
        deciles = pd.qcut(y_pred_proba, q=bins, duplicates='drop')
        
        lift_data = []
        baseline_fraud_rate = y_true.mean()
        
        for decile in sorted(deciles.unique(), reverse=True):
            mask = deciles == decile
            fraud_rate_in_decile = y_true[mask].mean()
            lift = fraud_rate_in_decile / baseline_fraud_rate if baseline_fraud_rate > 0 else 0
            percentage = mask.sum() / len(mask) * 100
            
            lift_data.append({
                'Decile': decile,
                'Fraud_Rate': fraud_rate_in_decile,
                'Lift': lift,
                'Percentage': percentage
            })
        
        return pd.DataFrame(lift_data)
