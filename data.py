from glob import glob
import pandas as pd
import re
from tqdm import tqdm
import json
from clean_df import clean_report_dfs
from split_data import train_test_split
from hdf5_to_images import create_images

def get_img_ids(img_fp: str):
    """
    The function `get_img_ids` takes a file path as input, searches for all JPG files in that directory,
    sorts the file names, and returns a list of image IDs.
    
    :param img_fp: file path where the images are located.
    :type img_fp: str
    :return: a list of image IDs (file names) of all the .jpg files
    in the specified directory `img_fp`, sorted in ascending order.
    """
    img_ids = glob('*.jpg', root_dir=img_fp)
    img_ids.sort()
    return img_ids

def generate_train_json(img_ids: list, train_set:set, t2020:pd.DataFrame, a2020:pd.DataFrame, data_fp: str, json_save_fp:str, question: str):
    """
    Generates a JSON file to training LLaVA with UCSD CXR data.
    
    :param img_ids: list of image IDs that you want to process in the function `generate_train_json`. 
    :type img_ids: list
    :param train_set: set of accession Ids in the training set
    :type train_set: set
    :param t2020: through 2020 reports dataframe
    :type t2020: pd.DataFrame
    :param a2020: After 2020 reports dataframe
    :type a2020: pd.DataFrame
    :param data_fp: 
    :type data_fp: str
    :param json_save_fp: filepath to save json
    :type json_save_fp: str
    :param question: string that will be appended to the prompt
    :type question: str
    """
    json_output = []
    c_t2020 = 0
    c_a2020 = 0

    for img_id in tqdm(img_ids):
        img_acc_id = str(re.search(r'[0-9]+', img_id)[0])
        patient_id = img_id.split('.')[0]

        # Find Report for image
        if img_acc_id in t2020.index and img_acc_id in train_set:
            c_t2020 += 1
            prompt = t2020.loc[img_acc_id, 'prompt'] + question
            answer = t2020.loc[img_acc_id, 'answer']
        elif img_acc_id in a2020.index and img_acc_id in train_set:
            c_a2020 += 1
            prompt = a2020.loc[img_acc_id, 'prompt'] + question
            answer = a2020.loc[img_acc_id, 'answer']
        else:
            continue

        # Create json entry for image
        patient_data = {
            "id": patient_id,
            "image": data_fp + img_id,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{prompt}"
                },
                {
                    "from": "gpt",
                    "value": f"{answer}"
                }
            ]
        }
        json_output.append(patient_data)

    with open(json_save_fp, 'w') as f:
        json.dump(json_output, f, indent=4)
    print(f'---Through 2020 Completed, Found {c_t2020} Patients---')
    print(f'---After 2020 Completed, Found {c_a2020} Patients---')
    print('Dumped JSON')

def generate_test_jsonl(img_ids: list, test_set:set, t2020:pd.DataFrame, a2020:pd.DataFrame, data_fp:str, jsonl_save_fp:str, question:str):
    """
    Generates a JSONL file to testing a fine-tuned LLaVA with UCSD CXR data.
    
    :param img_ids: list of image IDs that you want to process in the function `generate_train_json`. 
    :type img_ids: list
    :param test_set: set of accession Ids in the test set
    :type test_set: set
    :param t2020: through 2020 reports dataframe
    :type t2020: pd.DataFrame
    :param a2020: After 2020 reports dataframe
    :type a2020: pd.DataFrame
    :param data_fp: 
    :type data_fp: str
    :param json_save_fp: filepath to save json
    :type json_save_fp: str
    :param question: string that will be appended to the prompt
    :type question: str
    """
    with open(jsonl_save_fp, 'w') as f:
        for img_id in tqdm(img_ids):
            img_acc_id = str(re.search(r'[0-9]+', img_id)[0])
            patient_id = img_id.split('.')[0]
            if img_acc_id in t2020.index and img_acc_id in test_set:
                prompt = t2020.loc[img_acc_id, 'prompt'] + question
            elif img_acc_id in a2020.index and img_acc_id in test_set:
                prompt = a2020.loc[img_acc_id, 'prompt'] + question
            else:
                continue
            patient_data = {
                "question_id": patient_id,
                "image": data_fp + img_id,
                "text": f"{prompt}",
                "category": "",
            }
            f.write(json.dumps(patient_data) + "\n")
    print('Dumped JSON')

def generate_data():
    # Generate jpgs from h5 files
    h5_base = "/data/"
    jpg_dest_folder = "/data/UCSD_cxr/jpg"
    deid_keys_fp = "/data/UCSD_cxr/deid-keys.csv"
    create_images(h5_base, jpg_dest_folder, deid_keys_fp)

    # Generate Test Train split
    fake_id_fp = "/data/ucsd_cxr/falseids-accn.csv"
    train_fp = '/data/UCSD_cxr/train_accn.csv'
    test_fp = '/data/UCSD_cxr/test_accn.csv'
    train_test_split(fake_id_fp, train_fp, test_fp)    

    img_fp = '/data/UCSD_cxr/jpg/'
    img_ids = get_img_ids(img_fp)
    train_accn = pd.read_csv(train_fp)['Accession'].astype(str)
    test_accn = pd.read_csv(test_fp)['Accession'].astype(str)

    t2020, a2020 = clean_report_dfs()

    generic_question = 'Write a report for the given chest x-ray. It should contain a clear findings and impression section.  Be explicit for any abnormalities or normal findings.'
    context_question = '\nBased on AUTHOR and CLINICAL HISTORY, suppose you were a radiologist on X-RAY,  could you provide a detailed report from this chest X-ray?'
    generate_train_json(img_ids, set(train_accn), t2020, a2020, img_fp, './generic_prompt_train.json', generic_question)
    generate_test_jsonl(img_ids, set(test_accn), t2020, a2020, img_fp, './generic_prompt_test.jsonl', generic_question)

    generate_train_json(img_ids, set(train_accn), t2020, a2020, img_fp, './context_prompt_train.json', context_question)
    generate_test_jsonl(img_ids, set(test_accn), t2020, a2020, img_fp, './context_prompt_test.jsonl', context_question)

if __name__ == "__main__":
    generate_data()