import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import argparse
from transformers import pipeline
from transformers import pipeline
import logging 
from tqdm.auto import tqdm
# Import your classifier function and candidate_labels list here

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def classify_reports(reports, candidate_labels):
    '''produces probabilities for each candidate label for each report'''
    
    # ViktorDo/bart-base-finetuned-summaries-BioArxiv
    # maybe see how this does (not trusting that much to be honest)
    # if cuda is available, use it (device =0 for cuda and device = -1 for cpu)
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3", multilabel=True, device=device)
    results = classifier(reports, candidate_labels)
    structured_results = []
    for result in results:
        scores_dict = {label: score for label, score in zip(result['labels'], result['scores'])}
        scores_dict['report'] = result['sequence']
        structured_results.append(scores_dict)
    return structured_results

def process_dataframe(input_path, output_path, column_name, candidate_labels):
    # Load the DataFrame
    test_inference = pd.read_csv(input_path)

    # Extract the reports
    reports = list(test_inference[column_name].astype(str))

    # Classify reports
    structured_results = classify_reports(reports, candidate_labels)

    # Convert results to DataFrame
    results_df = pd.DataFrame(structured_results).round(2)
    columns = ['report'] + [col for col in results_df.columns if col != 'report']
    results_df = results_df[columns]
    
    results_df['id'] = test_inference['id']

    # Save to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Classification completed and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Classify radiologist reports and output to CSV.")
    parser.add_argument("input_path", help="Path to the input CSV file containing the radiologist reports.")
    parser.add_argument("output_path", help="Path to the output CSV file to save the classification results.")
    parser.add_argument("column_name", help="The column name containing the radiologist reports.")
    parser.add_argument("--candidate_labels", nargs="+", help="List of candidate labels for classification.", required=True)
    
    args = parser.parse_args()

    # Process the DataFrame and save results
    process_dataframe(args.input_path, args.output_path, args.column_name, args.candidate_labels)

if __name__ == "__main__":
    main()
