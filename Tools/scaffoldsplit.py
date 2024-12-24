import deepchem as dc
import pandas as pd
import numpy as np
import os


dataset_name = 'lipo'

for split_num in [1, 2, 3]:
    df = pd.read_csv(f'dataset/{dataset_name}.csv')

    input_data = f'dataset/{dataset_name}.csv'

    if dataset_name=='ADME3521':
        tasks = [col for col in df.columns if col not in ['SMILES', 'Internal ID', 'CollectionName', 'Vendor ID']]
        featurizer=dc.feat.CircularFingerprint(size=1024)
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field='SMILES', featurizer=featurizer)
    elif dataset_name=='bace':
        tasks = [col for col in df.columns if col not in ['mol', 'CID', 'Model']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'mol'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field='mol', featurizer=featurizer)
    elif dataset_name=='BBBP':
        tasks = [col for col in df.columns if col not in ['smiles', 'name']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'smiles'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)
    elif dataset_name=='clintox' or dataset_name=='sider':
        tasks = [col for col in df.columns if col not in ['smiles']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'smiles'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)
    elif dataset_name=='tox21':
        tasks = [col for col in df.columns if col not in ['smiles', 'mol_id']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'smiles'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)
    elif dataset_name=='esol':
        tasks = [col for col in df.columns if col not in ['Compound ID', 'smiles']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'smiles'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)
    elif dataset_name=='lipo':
        tasks = [col for col in df.columns if col not in ['CMPD_CHEMBLID', 'smiles']]
        featurizer = dc.feat.CircularFingerprint(size=1024)
        smiles_col = 'smiles'
        loader = dc.data.CSVLoader(tasks=tasks, smiles_field=smiles_col, featurizer=featurizer)

    dataset=loader.featurize(input_data)

    X, y, w, ids = dataset.X, dataset.y, dataset.w, dataset.ids
    perm = np.random.permutation(len(X))
    X, y, w, ids = X[perm], y[perm], w[perm], ids[perm]
    dataset = dc.data.NumpyDataset(X, y, w, ids)

    splitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = \
        (splitter.train_valid_test_split(
            dataset,
            frac_train=0.8,
            frac_valid=0.1,
            frac_test=0.1,
            seed=split_num
        )
    )
    train_dataset_1, valid_dataset_1, test_dataset_1 = \
        (splitter.train_valid_test_split(
            train_dataset,
            frac_train=0.75,
            frac_valid=0.125,
            frac_test=0.125,
            seed=split_num + 100
        )
    )

    train_ids = train_dataset.ids
    valid_ids = valid_dataset.ids
    test_ids = test_dataset.ids

    # 根据SMILES筛选原始DataFrame中对应的行
    train_df = df[df[smiles_col].isin(train_ids)]
    valid_df = df[df[smiles_col].isin(valid_ids)]
    test_df = df[df[smiles_col].isin(test_ids)]

    if not os.path.exists(f"{dataset_name}"):
        # 若不存在，则新建文件夹
        os.makedirs(f"{dataset_name}")
        print(f"Created folder {dataset_name}")

    if not os.path.exists(f"{dataset_name}/{dataset_name}_{split_num}"):
        # 若不存在，则新建文件夹
        os.makedirs(f"{dataset_name}/{dataset_name}_{split_num}")
        print(f"Created folder {dataset_name}/{dataset_name}_{split_num}")

    # 将分割后的数据集另存为CSV
    train_df.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/train_{split_num}.csv', index=False)
    valid_df.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/valid_{split_num}.csv', index=False)
    test_df.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/test_{split_num}.csv', index=False)

    train_ids_1 = train_dataset_1.ids
    valid_ids_1 = valid_dataset_1.ids
    test_ids_1 = test_dataset_1.ids

    train_df_1 = df[df[smiles_col].isin(train_ids_1)]
    valid_df_1 = df[df[smiles_col].isin(valid_ids_1)]
    test_df_1 = df[df[smiles_col].isin(test_ids_1)]

    train_df_1.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/train_{split_num}_1.csv', index=False)
    valid_df_1.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/valid_{split_num}_1.csv', index=False)
    test_df_1.to_csv(f'{dataset_name}/{dataset_name}_{split_num}/test_{split_num}_1.csv', index=False)

    print(train_dataset)