import pandas as pd
import numpy as np

def train_test_split(fake_id_fp: str, train_fp:str, test_fp:str):
    fakeid_accn = pd.read_csv(fake_id_fp).drop_duplicates()

    # get unique patients
    unq_fakeid = fakeid_accn['falseid'].unique()
    num_unq_fake = unq_fakeid.size
    np.random.seed(1)

    # split dataset
    shuffled_ids = np.random.choice(unq_fakeid, size=num_unq_fake, replace=False)
    train_size = 0.98
    split = int(train_size*num_unq_fake)
    train = shuffled_ids[:split]
    test = shuffled_ids[split:]

    train_accn = fakeid_accn[fakeid_accn['falseid'].isin(train)]
    test_accn = fakeid_accn[fakeid_accn['falseid'].isin(test)]
    train_accn.to_csv(train_fp)
    test_accn.to_csv(test_fp)