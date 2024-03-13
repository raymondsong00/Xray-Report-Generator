import pandas as pd
import numpy as np
import re

def df_setup(df_filepath: str):
    """
    The function `df_setup` reads a CSV file into a DataFrame, converts the 'AccessionId' column to
    string type, sets 'AccessionId' as the index, drops any rows with missing values, and returns the
    cleaned DataFrame.
    
    :param df_filepath: DataFrame filepath
    :type df_filepath: str
    :return: Dataframe with index as AccessionId strings 
    """
    df = pd.read_csv(df_filepath)
    df['AccessionId'] = df['AccessionId'].astype(str)
    df = df.set_index('AccessionId')
    df = df.dropna()
    return df

def extract_answer_with_unknown(report_text: pd.Series):
    """
    The function `extract_answer_with_unknown` extracts findings and impressions from a pandas Series of
    report text and concatenates them into a new answer.
    
    :param report_text: a pandas Series with xray report text
    :type report_text: pd.Series
    :return: The function `extract_answer_with_unknown` returns a new Series containing extracted
    findings and impressions from the input `report_text` Series. 
    """
    findings = report_text.str.extract(r'(?s)FINDINGS(.*?)(?=IMPRESSION:|Signed by:|\Z)')[0].str.strip()\
                        .fillna('').apply(lambda x: np.nan if len(x) < 10 else x)
    impression = report_text.str.extract(r'(?s)IMPRESSION:(.*?)(?=Signed by:|\Z)')[0].str.strip()\
                        .fillna('').apply(lambda x: np.nan if len(x) < 10 else x)

    findings = findings.apply(lambda x: 'FINDINGS:' + x if type(x) == str else x)
    impression = impression.apply(lambda x: 'IMPRESSION:' + x if type(x) == str else x)

    new_answers = pd.concat([findings, impression], axis=1)\
                    .apply(lambda x: '\n'.join(x.dropna()) if x.dropna().size > 0 else np.nan, axis=1)
    return new_answers
    

def drop_unknown(df: pd.DataFrame):
    """
    The function `drop_unknown` filters a DataFrame based on specific text patterns and removes unknown
    from answers.
    
    :param df: A pandas DataFrame containing a column named 'answer' with the generated answer for LLaVA
    :type df: pd.DataFrame
    :return: The function `drop_unknown` returns two new answers extracted from the input DataFrame
    `df`. The first new answer is extracted from rows where the 'answer' column contains the string
    'FINDINGS:\nUnknown', and the second new answer is extracted from rows where the 'answer' column
    contains the string 'IMPRESSION: Unknown'.
    """
    df_fu_bools= df['answer'].str.contains('FINDINGS:\nUnknown')
    df_iu_bools = df['answer'].str.contains('IMPRESSION: Unknown')
    df_fu = df[df_fu_bools]
    df_iu = df[df_iu_bools]
    new_answer_fu = extract_answer_with_unknown(df_fu['ReportText'])
    new_answer_iu = extract_answer_with_unknown(df_iu['ReportText'])
    return (new_answer_fu, new_answer_iu)
    

def contains_date(report):
    """
    The function `contains_date` checks if a given report contains a date in the mm/dd/yyyy format using
    regular expressions.
    
    :param report: report text
    :return: The function `contains_date` is returning a boolean value - `True` if the report contains a
    date in the mm/dd/yyyy format, and `False` if it does not.
    """
    date_pattern = r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    return True if re.search(date_pattern, report) else False

def remove_concurrent_supervision(report):
    """
    The function `remove_concurrent_supervision` removes the "CONCURRENT SUPERVISION" section and
    "Preliminary created by:" from a report while keeping the "IMPRESSION" section.
    
    :param report: report text
    :return: report text without concurrent supervision
    """
    cleaned_report = re.sub(r"CONCURRENT SUPERVISION:.*?IMPRESSION:", "IMPRESSION:", report, flags=re.DOTALL)
    return cleaned_report

def clean_report_dfs():
    """
    The function `clean_report_dfs` processes two DataFrames by dropping unknown values, removing
    duplicates, filtering out reports with 'Unknown' or date references, and applying specific cleaning
    functions to the 'answer' column.
    :return: The function `clean_report_dfs` is returning two cleaned DataFrames `t2020` and `a2020`.
    """
    t2020 = df_setup('/data/UCSD_cxr/through2020_dropna_formatted.csv')
    a2020 = df_setup('/data/UCSD_cxr/after2020_dropna_formatted.csv')

    new_answer_t2020_fu, new_answer_t2020_iu = drop_unknown(t2020)
    new_answer_a2020_fu, new_answer_a2020_iu = drop_unknown(a2020)

    t2020_filled_missing = new_answer_t2020_fu.combine_first(new_answer_t2020_iu).dropna()
    a2020_filled_missing = new_answer_a2020_fu.reset_index().groupby('AccessionId').last()[0] \
        .combine_first(new_answer_a2020_iu.groupby('AccessionId').last().dropna()).dropna()

    # Update answer column without unknowns
    t2020.loc[t2020_filled_missing.index, 'answer'] = t2020_filled_missing
    a2020.loc[a2020_filled_missing.index, 'answer'] = a2020_filled_missing

    t2020 = t2020.reset_index().drop_duplicates(subset='AccessionId').set_index('AccessionId')
    a2020 = a2020.reset_index().drop_duplicates(subset='AccessionId').set_index('AccessionId')

    # Remove anything else with Unknown
    t2020 = t2020[~t2020['answer'].str.contains('Unknown')]
    a2020 = a2020[~a2020['answer'].str.contains('Unknown')]
        
    # Drop reports with dates which are references to previous images    
    t2020 = t2020[~t2020['answer'].apply(contains_date)]
    a2020 = a2020[~a2020['answer'].apply(contains_date)]

    # Apply the cleaning function to each report in the 'answer' column
    a2020['answer'] = a2020['answer'].apply(remove_concurrent_supervision)
    t2020['answer'] = t2020['answer'].apply(remove_concurrent_supervision)

    # drop the stragglers (there is only 3 anyway)
    t2020 = t2020[~t2020['answer'].str.contains('CONCURRENT')]
    a2020 = a2020[~a2020['answer'].str.contains('CONCURRENT')]

    return t2020, a2020