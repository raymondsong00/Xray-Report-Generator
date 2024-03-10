
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import argparse
from transformers import pipeline
import evaluate
import os
from matplotlib.backends.backend_pdf import PdfPages
import logging
from fpdf import FPDF
import json

import seaborn as sns



def break_convos(df, col1, col2):
    '''takes in conversations column and breaks it up into separate columns for data frame'''
    ser = df['conversations']
    prompt = pd.json_normalize(ser.apply(lambda x: x[0]))['value']
    impressions = pd.json_normalize(ser.apply(lambda x: x[1]))['value']

    df[col1] = prompt
    df[col2] = impressions

    df = df.drop(axis=1, columns=['conversations'])

    return df

def make_image_path(row, img_dir='/data/UCSD_cxr/jpg/'):
    if 'id' in list(row.index):
        return img_dir + row['phonetic_id'] + '_' + str(row['id']) + '.jpg'
    # else it's the index of the row
    return img_dir + row['phonetic_id'] + '_' + str(row.name) + '.jpg'

def read_image(image_path):
    # Read the image
    img = mpimg.imread(image_path)

    # Display the image
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Turn off axis numbers
    plt.show()

def select_random_img():
    idx = np.random.randint(merged_df.shape[0])
    row = merged_df.iloc[idx]

    read_image(row['image'])
    print(f"id: {row['id']}\nradiologist report: {row['radiologist_report']}\nllava_prompt: {row['llava_prompt']}\nimpression: {row['llava_impression']}")


def read_and_process(test_inference_path, train_data_path, test_radiologist_answers_path, t2020_path='/data/UCSD_cxr/through2020_dropna_formatted.csv', a2020_path='/data/UCSD_cxr/after2020_dropna_formatted.csv'):   
    '''
    Ingest and process data from llava fine tune and test set inference process 
    
    See intersection.ipynb for generation of data files
    
    Args:
    test_inference_path (str): path to test inference json file
    train_data_path (str): path to train data json file
    test_radiologist_answers_path (str): path to test radiologist answers csv file
    t2020_path (str): path to thorugh 2020 csv file
    a2020_path (str): path to after 2020 csv file
    
    Returns:
    test_inference (pd.DataFrame): test inference data with new columns
    train_df (pd.DataFrame): train data with new columns
    '''
    
    # read in jsons produced from llava fine tune and break them into useful columns and save as csv files 
    test_inference = pd.read_json(test_inference_path, lines=True)
    train_df = pd.read_json(train_data_path).drop(axis=1, columns='image')
    train_df = break_convos(train_df, 'prompt', 'radiologist_report')

    csv_t2020_df = pd.read_csv(t2020_path).dropna().drop_duplicates()
    csv_a2020_df = pd.read_csv(a2020_path).dropna().drop_duplicates()
    csv_a2020_df = csv_a2020_df[list(csv_t2020_df.columns)]

    # this is deprecated I think
    # all_data_df = pd.concat([csv_t2020_df, csv_a2020_df], axis=0).drop_duplicates()

    # all_data_df[['radiologist_findings', 'radiologist_impression']] = all_data_df['ReportText'].str.extract(r'(?s)(FINDINGS.*?)(IMPRESSION:.*?)(?:CONCURRENT|$)')
    # all_data_df = all_data_df.dropna(subset=['radiologist_findings', 'radiologist_impression'])

    # all_data_df['radiologist_report'] = all_data_df['radiologist_findings'] + ' ' + all_data_df['radiologist_impression']

    test_inference['id'] = test_inference['question_id'].str.extract(r'\w+_(\d+)').astype(int)
    test_inference['phonetic_id'] = test_inference['question_id'].str.extract(r'^(\w+)_').astype(str)

    train_df['id'] = train_df['id'].str.extract(r'\w+_(\d+)').astype(int)

    test_inference.rename(columns={'text': 'llava_report'}, inplace=True)

    test_inference = test_inference[['id', 'phonetic_id', 'prompt', 'llava_report']]


    extract1 = train_df['prompt'].str.extract(r'(?s)AUTHOR:(.*)CLINICAL HISTORY:')[0].str.strip()
    extract2 = train_df['prompt'].str.extract(r'(?s)AUTHOR:\s(\w+,\s\w+)')[0].str.strip()

    extract1.fillna(extract2, inplace=True)
    train_df['author'] = extract1.fillna('unknown')

    extract1 = test_inference['prompt'].str.extract(r'(?s)AUTHOR:(.*)CLINICAL HISTORY:')[0].str.strip()
    extract2 = test_inference['prompt'].str.extract(r'(?s)AUTHOR:\s(\w+,\s\w+)')[0].str.strip()
    extract1.fillna(extract2, inplace=True)
    test_inference['author'] = extract1.fillna('unknown')

    test_inference_radiologist_answers = pd.read_csv(test_radiologist_answers_path)
    test_inference = test_inference[['id', 'phonetic_id', 'llava_report']].merge(test_inference_radiologist_answers, left_on='id', right_on='AccessionId', how='inner')[['id', 'phonetic_id', 'Author', 'prompt', 'llava_report', 'answer']]
    test_inference.rename(columns={'Author': 'author', 'answer': 'radiologist_report'}, inplace=True)

    test_inference['llava_findings'] = test_inference['llava_report'].str.extract(r'(?s)(FINDINGS:.*?)(?:IMPRESSION:|$)')
    test_inference['llava_impression'] = test_inference['llava_report'].str.extract(r'(?s)(IMPRESSION:.*?)(?:CONCURRENT|$)')

    test_inference['radiologist_findings'] = test_inference['radiologist_report'].str.extract(r'(?s)(FINDINGS:.*?)(?:IMPRESSION:|$)')
    test_inference['radiologist_impression'] = test_inference['radiologist_report'].str.extract(r'(?s)(IMPRESSION:.*?)(?:CONCURRENT|$)')

    train_df['radiologist_findings'] = train_df['radiologist_report'].str.extract(r'(?s)(FINDINGS:.*?)(?:IMPRESSION:|$)')
    train_df['radiologist_impression'] = train_df['radiologist_report'].str.extract(r'(?s)(IMPRESSION:.*?)(?:CONCURRENT|$)')

    # eliminate duplicate rows
    # TODO: fix this in the upstream data process (means we may not be eliminating all duplicates based on id) 
    original_length_test = len(test_inference)
    original_length_train = len(train_df)

    test_inference = test_inference.drop_duplicates(subset=['id'])
    train_df = train_df.drop_duplicates(subset=['id'])

    num_dropped_test = original_length_test - len(test_inference)
    num_dropped_train = original_length_train - len(train_df)

    # log on how many duplicates were dropped TODO

    # record and log how many missing values there are
    missing_rows = test_inference[test_inference.isna().any(axis=1)]
    missing_columns = missing_rows.columns[missing_rows.isna().any()].tolist()
    missing_rows_filtered = missing_rows[missing_columns]
    # missingness count by column
    missing_counts = dict(missing_rows_filtered.isna().sum() )
    # TODO log this information or save to json file 

    # record and log missing values for the train data
    missing_rows = train_df[train_df.isna().any(axis=1)]
    missing_columns = missing_rows.columns[missing_rows.isna().any()].tolist()
    missing_rows_filtered = missing_rows[missing_columns]
    # missingness count by column
    missing_counts = dict(missing_rows_filtered.isna().sum()) 

    # fill in missing values with empty string 
    test_inference.fillna('', inplace=True)
    train_df.fillna('', inplace=True)

    test_inference = test_inference[['id', 'phonetic_id', 'author', 'prompt', 'llava_report', 'llava_findings', 'llava_impression', 'radiologist_report', 'radiologist_findings', 'radiologist_impression']]
    train_df = train_df[['id', 'author', 'prompt', 'radiologist_report', 'radiologist_findings', 'radiologist_impression']]
    
    return test_inference, train_df

def get_sim_scores(col1, col2, sent_model):
    col1 = col1.fillna('')
    col2 = col2.fillna('')
    
    col1 = col1.str.replace('FINDINGS:', '')
    col2 = col2.str.replace('IMPRESSION:', '')
    
    col1 = col1.to_numpy()
    col2 = col2.to_numpy()
    
    embeddings1 = sent_model.encode(col1)
    embeddings2 = sent_model.encode(col2)
    
    # Normalize the embeddings
    embeddings1_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2_norm = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarity_scores = np.diag(embeddings1_norm @ embeddings2_norm.T)
    
    return similarity_scores


def sample_by_bin(df, col, bin_index):
    '''
    col -- name of column to get similarities from
    bin_index -- a number from 1 to 11 (probably)
    
    returns a random row from the specified bin'''
    sim_scores = df[col]
    bin_indices = np.digitize(sim_scores, bins=np.histogram_bin_edges(sim_scores))
    bools = bin_indices == bin_index
    filtered = sim_scores[bools]
    df_subset = df[bools]
    
    i = np.random.randint(0, high=len(filtered))
    return df_subset.iloc[i]

def compact_report(report_string):
    # Step 1: Replace new line characters with a space
    report_string = report_string.replace('\n', ' ')
    
    # Step 2: Strip extra spaces
    report_string = ' '.join(report_string.split())
    
    return report_string


def wrap_text(text, max_line_length):
    words = text.split()
    lines = []
    current_line = ''
    
    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += ' ' + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    
    # Append the last line if it's non-empty
    if current_line:
        lines.append(current_line)
    
    return '\n'.join(lines)

def print_row_nicely(row, img_dir='/data/UCSD_cxr/jpg/', l=50):
    """
    This function takes a row of data (as a dictionary) and prints it out in a formatted manner.
    
    Parameters:
    row (dict): A dictionary representing a row of data, where keys are column names.
    img_dir (str): The directory where the images are stored.
    l (int): The maximum line length for wrapping text.
    """
    read_image(make_image_path(row, img_dir))
    print("Row Data Overview:")
    print(f"ID:".ljust(25) + f"{row.name}")
    print("-" * l)  # Print a divider for better readability
    for key, value in row.items():
        value = wrap_text(str(value), l)
        
        print(f"{key}:".ljust(25) + f"{value}")
    print("-" * 50)  # End with a divider for a clean look

def calculate_text_lengths(test_inference):
    see_impression_bools = test_inference['llava_findings'].str.lower().str.contains('see impression') | test_inference['llava_findings'].str.lower().str.contains('unknown') | (test_inference['llava_findings'] == '')
    see_impression = test_inference['llava_findings'].str.lower()[see_impression_bools]
    see_impression_lens = see_impression[see_impression_bools].str.len()

    test_inference['is_si_l'] = see_impression_bools 
    test_inference['is_si_r'] = test_inference['radiologist_findings'].str.lower().str.contains('see impression') | test_inference['radiologist_findings'].str.lower().str.contains('unknown') | (test_inference['radiologist_findings'] == '')

    llava_impression_lengths = test_inference['llava_impression'].str.len()
    llava_findings_lengths = test_inference['llava_findings'].str.len()
    radiologist_impression_lengths = test_inference['radiologist_impression'].str.len()
    radiologist_findings_lengths = test_inference['radiologist_findings'].str.len()

    test_inference['lf_len'] = llava_findings_lengths
    test_inference['li_len'] = llava_impression_lengths
    test_inference['rf_len'] = radiologist_findings_lengths
    test_inference['ri_len'] = radiologist_impression_lengths


def plot_lengths(test_inference, save_dir, threshold=1000):
    llava_findings_lengths = test_inference['lf_len'] 
    llava_impression_lengths = test_inference['li_len']
    radiologist_findings_lengths = test_inference['rf_len']
    radiologist_impression_lengths = test_inference['ri_len']

    # Llava Findings Lengths
    fig1 = plt.figure()
    llava_findings_lengths[llava_findings_lengths < threshold].plot(kind='hist')
    plt.title('Llava Findings Section Lengths')
    plt.axvline(np.median(llava_findings_lengths[llava_findings_lengths < threshold]), color='red')
    save_path = os.path.join(save_dir, 'llava_findings_lengths.png')
    plt.savefig(save_path)
    #plt.clf()

    # Radiologist Findings Lengths
    fig2 = plt.figure()
    radiologist_findings_lengths.plot(kind='hist')
    plt.title('Radiologist Findings Section Lengths')
    plt.axvline(np.median(radiologist_findings_lengths), color='red')
    save_path = os.path.join(save_dir, 'radiologist_findings_lengths.png')
    plt.savefig(save_path)
    #plt.clf()

    # Llava Impression Lengths
    fig3 = plt.figure()
    llava_impression_lengths[llava_impression_lengths < threshold].plot(kind='hist')
    plt.title('Llava Impression Section Lengths')
    plt.axvline(np.median(llava_impression_lengths[llava_impression_lengths < threshold]), color='red')
    save_path = os.path.join(save_dir, 'llava_impression_lengths.png')
    plt.savefig(save_path)
    #plt.clf()

    # Radiologist Impression Lengths
    fig4 = plt.figure()
    radiologist_impression_lengths.plot(kind='hist')
    plt.title('Radiologist Impression Section Lengths')
    plt.axvline(np.median(radiologist_impression_lengths), color='red')
    save_path = os.path.join(save_dir, 'radiologist_impression_lengths.png')
    plt.savefig(save_path)
    #plt.clf() 

# def plot_sim_scores(test_inference, col='similarity', save_path=None):
#     '''creates histogram plot of similarity scores from the specified column and saves the plot if a save path is provided'''

#     fig, ax = plt.subplots()  # Create a figure and axes object
#     ax.grid(False)
#     test_inference[col].hist(ax=ax)  # Plot histogram on the axes
#     ax.set_title(f'{col} scores')
#     ax.axvline(test_inference[col].median(), color='red', label='Median')
#     ax.set_xlabel(f'{col} Score (0-1)')
#     ax.set_ylabel('Counts')
#     ax.legend()

#     if save_path is not None:
#         # Ensure the directory exists
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         # Save the plot
#         plt.savefig(save_path)
#         plt.close(fig)  # Close the plot to free up memory
#     else:
#         plt.show()  # Show the plot if no save path is provided

def plot_sim_scores(test_inference, col='similarity', save_path=None):
    '''creates a more visually appealing histogram plot of similarity scores from the specified column and saves the plot if a save path is provided'''

    sns.set(style="whitegrid")  # Set the seaborn style for more appealing plots

    fig, ax = plt.subplots()  # Create a figure and axes object
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Enhance grid appearance
    sns.histplot(test_inference[col], ax=ax, binwidth=0.05, kde=True, color='skyblue', edgecolor='black')  # Use seaborn for histogram
    
    # Highlight the median value
    median_value = test_inference[col].median()
    ax.axvline(median_value, color='red', label=f'Median: {median_value:.2f}', linestyle='--')

    # Improve the aesthetics
    ax.set_title(f'Context Embedded Prompt {col.capitalize()} Scores Distribution', fontsize=16)
    ax.set_xlabel(f'{col.capitalize()} Score (0-1)', fontsize=14)
    ax.set_ylabel('Counts', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend()

    # Tweak the layout for a more compact and aesthetically pleasing appearance
    plt.tight_layout()

    if save_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save the plot
        plt.savefig(save_path, dpi=300)  # Use a higher DPI for better image quality
        plt.close(fig)  # Close the plot to free up memory
    else:
        plt.show()  # Show the plot if no save path is provided
        



def sim_score_by_author(test_inference, author, save_dir, sim_col='similarity'):
    '''creates a histogram of that author's sim scores'''
    plt.figure(figsize=(10, 8))
    filtered = test_inference[test_inference['author'] == author]
    filtered[sim_col].plot(kind='hist')
    plt.title(f'{author}')
    plt.xlabel(f'{sim_col} scores')
    
    last_name = author.split(', ')[0]
    save_path = os.path.join(save_dir, f'{last_name}-{sim_col}')
    plt.savefig(save_path)
    plt.clf()
    

def plot_author_medians(test_inference, save_dir, sim_col='similarity', k=10):
    '''plots a bar plot with the median similarity for each author'''
    author_val_counts = test_inference['author'].value_counts()
    top_authors = set(author_val_counts[0:k].index)
    
    fig = plt.figure()
    test_inference.groupby('author')[sim_col].median().loc[list(top_authors)].sort_values().plot(kind='bar')
    plt.title(f'Median Similarity Score for top {str(k)} authors')
    plt.xlabel('Author')
    plt.ylabel(f'{sim_col} author median')
    
    save_path = os.path.join(save_dir, f'top-{str(k)}-{sim_col}-author-medians')
    return fig


def rouge_scores(test_inference, save_dir, reference_col='radiologist_report', candidate_col='llava_report', rouge_cols = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']):
    ''' uses rouge score (a score to calculate the accuracy of text to text models) of a candidate (text generated by the 
       model) vs a reference (the ground truth persay from the radiologist) '''
       # calculate scores
    rouge = evaluate.load('rouge')

    # add to data frame
    test_inference[rouge_cols] = test_inference.apply(lambda x: rouge.compute(predictions=[x[candidate_col]], references=[x[reference_col]]), 
                            axis=1, result_type='expand').round(2)
    
    # plot rouge columns
    for col in rouge_cols:
        plt.figure()
        test_inference[col].plot.hist(alpha=0.5, bins=20, title=f'Histogram of {col} candidate: {reference_col} vs {candidate_col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        save_path = os.path.join(save_dir, f'{col}-{reference_col}-{candidate_col}')
        plt.savefig(save_path)
        plt.clf()
        
        
# Compile plots into a PDF
def compile_plots_to_pdf(output_dir):
    title = output_dir.split('/')[-2]
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Setting the title font: Arial bold 15
    pdf.set_font('Arial', 'B', 15)
    # Calculate width of title and position
    title_width = pdf.get_string_width(title) + 6
    document_width = pdf.w - 2*pdf.l_margin
    # Set position for the title to be center aligned
    start_x = (document_width - title_width) / 2
    pdf.set_x(start_x)
    
    # Adding the title
    pdf.cell(title_width, 10, title, 0, 1, 'C')
    
    # Add a line break after the title
    pdf.ln(10)
    
    # Reset font for the plots
    pdf.set_font('Arial', '', 12)
    
    for plot in os.listdir(output_dir):
        if plot.endswith('.png'):
            pdf.add_page()
            pdf.image(os.path.join(output_dir, plot), x=10, y=10, w=180)
            
    pdf.output(os.path.join(output_dir, 'plots.pdf'), 'F')
    

# descriptive statistics functions for comparing datasets and results
# focus on test inference for this one, can create statistics for the others later
def descriptive_stats(test_inference, train_df, save_dir, k=10, d=2):
    '''generate descriptive statistics about model inference data
    and saves them to a json file in a human friendly, readable format'''
    json_dict = dict()
    
    top_k_test = test_inference['author'].value_counts(normalize=True, sort=True).round(d).iloc[0:k]
    top_k_train = train_df['author'].value_counts(normalize=True, sort=True).round(d).iloc[0:k]
    
    json_dict['author_counts_test'] = dict(top_k_test)
    json_dict['author_counts_train'] = dict(top_k_train)
    
    json_dict['test_size'] = test_inference.shape[0]
    json_dict['train_size'] = train_df.shape[0]
    
    # see impression llava proportion
    json_dict['si_l_prop'] = test_inference['is_si_l'].mean().round(d)
    # see impression radiologist proportion for comparison
    json_dict['si_r_prop'] = test_inference['is_si_r'].mean().round(d)
    
    json_dict['lf_len_median'] = test_inference['lf_len'].median().round()
    json_dict['li_len_median'] = test_inference['li_len'].median().round()
    
    json_dict['rf_len_median'] = test_inference['rf_len'].median().round()
    json_dict['ri_len_median'] = test_inference['ri_len'].median().round()
    
    # json_dict['rouge1_median ']= test_inference['rouge1'].median().round(d)
    # json_dict['rouge2_median'] = test_inference['rouge2'].median().round(d)
    # json_dict['rougeL_median'] = test_inference['rougeL'].median().round(d)
    # json_dict['rougeLsum_median'] = test_inference['rougeLsum'].median().round(d)
    
    
    json_file_path = f"{save_dir}/descriptive_stats.json"
    with open(json_file_path, 'w') as json_file:
        json.dump(json_dict, json_file, indent=4)
        
    return json_dict
    

def main(output_path, test_inference_path, train_data_path):
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
    
    plot_path = os.path.join(output_path, 'plots')
    data_path = os.path.join(output_path, 'data')
    
    # create folders to save in if they don't exist yet
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # can add argument for sentence transformer
    # The above code in Python is creating a sentence transformer model using the 'all-mpnet-base-v2'
    # pre-trained model. This model is used for converting sentences into numerical vectors that
    # capture the semantic meaning of the text.
    #
    # sent_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # this is the same model as above, but is fine tuned on biomedical data
    sent_model = SentenceTransformer('FremyCompany/BioLORD-2023')

    # returns the test inference csv, with cleaned up columns
    # TODO: create argument for shell script and add to argparse, main function
    test_inference_radiologist_answers_path = './raymond/030424_test_patient_finding_impression_answers.csv'
    
    test_inference, train_df = read_and_process(test_inference_path, train_data_path, test_inference_radiologist_answers_path)
    logging.warning('Data Loaded and Processed')
    
    # create report similarity scores with sentence transformer
    test_inference['similarity'] = get_sim_scores(test_inference['llava_report'], test_inference['radiologist_report'], sent_model)
    
    # keep select portion of columns
    test_inference = test_inference[['id',
                                    'phonetic_id',
                                    'author',
                                    'llava_findings',
                                    'llava_impression',
                                    'llava_report',
                                    'radiologist_report',
                                    'radiologist_findings',
                                    'radiologist_impression',
                                    'similarity']]
    
    # create more similarity scores between impressions, findings
    test_inference['impression_similarity'] = get_sim_scores(test_inference['llava_impression'], test_inference['radiologist_impression'], sent_model)
    test_inference['findings_similarity'] = get_sim_scores(test_inference['llava_findings'], test_inference['radiologist_findings'], sent_model)
    
    logging.warning('Embeddings Similarity Scores Created')
    
    # plot text lenghts 
    calculate_text_lengths(test_inference)
    plot_lengths(test_inference, plot_path)
    
    # plot report similarity scores
    #plot_sim_scores(test_inference, plot_path)
    # TODO capture return and save this plot 
    plot_sim_scores(test_inference, col='similarity', save_path=plot_path + '/report_similarity.png')
    #plot impression similarity scores
    # TODO
    plot_sim_scores(test_inference, col='impression_similarity', save_path=plot_path + '/impression_similarity.png')
    # plot findings similarity scores 
    # TODO 
    plot_sim_scores(test_inference, col='findings_similarity', save_path=plot_path + '/findings_similarity.png')
    
    # TODO: make better matplotlib plots for the and put them onto one page in grid 
    
    # plot similarity scores for Albert
    sim_score_by_author(test_inference, 'Hsiao, Albert', plot_path)
    
    # plot author similarity medians (kwarg k for top k authors to plot)
    plot_author_medians(test_inference, plot_path)
    logging.warning('Quick Plots Completed')
    
    #rouge_scores(test_inference, plot_path)
    logging.warning('Rouge Scores Calculated, Plotted')
    
    # create pdf
    test_inference.to_csv(os.path.join(data_path, 'test_inference.csv'), index=False)
    #test_questions.to_csv(os.path.join(data_path, 'test_questions.csv'), index=False)
    train_df.to_csv(os.path.join(data_path, 'train_data.csv'), index=False)
    logging.warning('Processed CSVs saved')
    compile_plots_to_pdf(plot_path)
    logging.warning('Plots PDF generated')
    descriptive_stats(test_inference, train_df, plot_path)
    logging.warning('Stats created')
    logging.warning('Eval Complete!')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate model performance and generate plots.")
    parser.add_argument("--output_path", help="output where resulting plots and data will be saved")
    
    # Adding optional keyword arguments
    parser.add_argument("--test_inference_path", help="path to jsonl file that llava performed inference on")
    #parser.add_argument("--test_questions_path", help="path to jsonl file that llava received for questions")
    parser.add_argument("--train_data_path", help="path to json file with llava finetune training data")

    
    args = parser.parse_args()
    
    main(args.output_path, args.test_inference_path, args.train_data_path)

