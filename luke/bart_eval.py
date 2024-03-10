import pandas as pd
import matplotlib.pyplot as plt
import os 
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.colors import LogNorm

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def read_scores_data(llava_path, radiologist_path):
    '''read in llava and radiologist scores and reports
    
    returns:
    llava_scores: pandas df
    radiologist_scores: pandas df
    llava_reports: pandas series
    radiologist_reports: pandas series'''
    
    # scores df
    llava_df = pd.read_csv(llava_path)
    radiologist_df = pd.read_csv(radiologist_path) 
    
    llava_df.drop_duplicates(subset=['id'], inplace=True)
    llava_df.set_index('id', inplace=True)
    radiologist_df.drop_duplicates(subset=['id'], inplace=True)
    radiologist_df.set_index('id', inplace=True)
    
    llava_scores  = llava_df.select_dtypes(include='number').round(2)
    radiologist_scores  = radiologist_df.select_dtypes(include='number').round(2)
    
    order_cols = list(radiologist_scores.mean().index)
    
    llava_scores = llava_scores[order_cols]
    radiologist_scores = radiologist_scores[order_cols]
    
    return llava_scores, radiologist_scores, llava_df['report'], radiologist_df['report']
    
    
def generate_labels(row, threshold=0.5):
    '''Replace scores with labels on whether conditions are present or not (binary),
    based on some threshold.
    
    Apply on axis=1 of data frame with scores only (no report column).
    
    Args:
    row: pandas series'''
    
    max_col = row.idxmax()
    max_val = row[max_col]
    
    # Check if 'no abnormalities mentioned' column exists
    no_abnormalities_column = 'no abnormalities mentioned' in row.index
    
    if no_abnormalities_column and max_col == 'no abnormalities mentioned':
        row[max_col] = 1
        row = row.where(row >= max_val, 0, inplace=False)
    else:
        row = row.where(row >= threshold, 0, inplace=False)
        row = row.mask(row >= threshold, 1, inplace=False)
        
        # If other conditions are true, set 'no abnormalities mentioned' to 0, if the column exists
        if no_abnormalities_column:
            row['no abnormalities mentioned'] = 0
        
    return row.astype(bool)

# Function to plot confusion matrix with a logarithmic color scale
def plot_confusion_matrix_log_scale(true_labels, predicted_labels, condition, ax):
    cm = confusion_matrix(true_labels, predicted_labels, labels=[True, False])
    # Apply log normalization to the color scale (add 1 to avoid log(0))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', norm=LogNorm(vmin=1, vmax=cm.max()), cbar=False, ax=ax)
    ax.set_title(f'Confusion Matrix: {condition}')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_xticklabels(['True', 'False'])
    ax.set_yticklabels(['True', 'False'])

def plot_confusion_matrices(df_true, df_predicted, sav_dir=None):
    conditions = df_predicted.columns

    # Calculate number of rows needed for subplots (each row contains 2 plots)
    n_rows = np.ceil(len(conditions) / 2).astype(int)

    # Set up the matplotlib figure and axes
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, n_rows*5))  # Width is for two columns
    axes = axes.flatten()  # Flatten the array to easily iterate over it

    # Generate and plot confusion matrix for each condition
    for i, condition in enumerate(conditions):
        true_labels = df_true[condition].astype('bool')
        predicted_labels = df_predicted[condition].astype('bool')
        plot_confusion_matrix_log_scale(true_labels, predicted_labels, condition, axes[i])

    # If the number of conditions is odd, remove the last ax
    if len(conditions) % 2 != 0:
        fig.delaxes(axes[-1])

    # Adjust layout
    plt.tight_layout()
    
    if sav_dir:
        plt.savefig(os.path.join(sav_dir, 'confusion_matrices.png'))
    
    # Return the figure object
    return fig

def label_data_frame(score_df, threshold=0.5):
    '''replace bart scores with labels on whether conditions are present or not (binary), 
    based on some threshold
    
    args:
    llava_scores: pandas df
    radiologist_scores: pandas df
    threshold: float'''
    
    label_df = score_df.apply(generate_labels, axis=1, threshold=threshold)
    
    return label_df
    
def plot_roc_curves(true_labels_df, predicted_scores_df, sav_dir=None):
    # Calculate ROC curve and ROC area for each condition
    fpr = dict()  # False Positive Rate
    tpr = dict()  # True Positive Rate
    roc_auc = dict()  # Area Under the Curve (AUC)
    
    conditions = predicted_scores_df.columns
    for i, condition in enumerate(conditions):
        fpr[condition], tpr[condition], _ = roc_curve(true_labels_df[condition], predicted_scores_df[condition])
        roc_auc[condition] = auc(fpr[condition], tpr[condition])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    
    for condition in conditions:
        plt.plot(fpr[condition], tpr[condition], label=f'{condition} (area = {roc_auc[condition]:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal dashed line for reference
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Generic Prompt')
    plt.legend(loc="lower right")
    plt.show()
    
    if sav_dir:
        plt.savefig(os.path.join(sav_dir, 'roc_curves_final.png'), bbox_inches='tight')


def main(llava_scores_path, radiologist_scores_path, output_path):
    llava_scores, radiologist_scores, lr, rr = read_scores_data(llava_scores_path, radiologist_scores_path)
    
    rr_labels = label_data_frame(radiologist_scores)
    llava_labels = label_data_frame(llava_scores)
    
    plot_confusion_matrices(rr_labels, llava_labels, sav_dir=output_path)
    
    plot_roc_curves(rr_labels, llava_scores, sav_dir=output_path)
    
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Report Through Labelling Data")
    parser.add_argument("--llava_scores_path", help="Path to the input CSV file containing the llava scores.")
    parser.add_argument("--radiologist_scores_path", help="Path to the input CSV file containing the radiologist scores.")
    parser.add_argument("--output_path", help="output where resulting plots and data will be saved")

    
    args = parser.parse_args()
    
    main(args.llava_scores_path, args.radiologist_scores_path, args.output_path)

