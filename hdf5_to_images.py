import h5py
import numpy as np
import pandas as pd
from PIL import Image
import os
from glob import glob
from tqdm import tqdm

def get_accession_ids(h5_keys, deid_df):
    imgs_keys = pd.DataFrame([key.split("__") for key in h5_keys], columns=["Phonetic ID", "Study Key", "Photo_ID"])
    first_image = imgs_keys[imgs_keys["Photo_ID"] == "0"]
    merge_study_key = first_image.merge(deid_df[["Study Key", "Accession Number"]], left_on="Study Key", right_on="Study Key")
    return merge_study_key

def convert_h5_to_jpg(image_data, jpg_file_path):
    image_data = ((image_data - image_data.min()) / (image_data.ptp()) * 255).astype(np.uint8)
    img = Image.fromarray(image_data)
    img.save(jpg_file_path)
    
def create_images(h5_base:str, jpg_dest_folder: str, deid_keys_fp):
    if not os.path.exists(jpg_dest_folder):
        os.makedirs(jpg_dest_folder)
    h5_files = glob(h5_base + '/*.hdf5')

    deid_keys_df = pd.read_csv(deid_keys_fp).drop_duplicates()

    for f_name in tqdm(h5_files):
        h5_data = h5py.File(f_name)
        keys = list(h5_data.keys())
        merged_df = get_accession_ids(keys, deid_keys_df)
        for i, (p_id, s_id, photo_id, acc_num) in merged_df.iterrows():
            dataset = h5_data[f"{p_id}__{s_id}__{photo_id}"][:]
            jpg_file_path = os.path.join(jpg_dest_folder, f'{p_id}_{acc_num}.jpg')
            convert_h5_to_jpg(dataset, jpg_file_path)

