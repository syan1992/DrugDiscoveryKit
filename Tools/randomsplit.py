from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pin1 = pd.read_csv('C:\\Users\\syan\\Documents\\record\\phd\\2024\\molecular property prediction\\pin1\\AID_504891_datatable_all.csv')

'''
new_pin1 = pin1[['PUBCHEM_EXT_DATASOURCE_SMILES', 'PUBCHEM_ACTIVITY_OUTCOME', 'Potency', 'Efficacy']]
new_pin1['y'] = new_pin1['PUBCHEM_ACTIVITY_OUTCOME'].apply(lambda x: 1 if x == 'Active' else 0)
new_pin1['Potency'] = pd.to_numeric(new_pin1['Potency'], errors='coerce')
new_pin1['Efficacy'] = pd.to_numeric(new_pin1['Efficacy'], errors='coerce')
new_pin1['P_E'] = (new_pin1['Potency'] / new_pin1['Efficacy']) ** 0.5
new_pin1.to_csv('C:\\Users\\syan\\Documents\\record\\phd\\2024\\molecular property prediction\\pin1\\pin1_dataset.csv', index=False)
new_pin1 = new_pin1.dropna()
new_pin1 = new_pin1[new_pin1['P_E'] < 0.95]
'''

X_train, X_test, y_train, y_test = train_test_split(new_pin1, new_pin1['y'], test_size=0.2） #, stratify=new_pin1['y'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

# 将 y_train 和 X_train 合并
train = X_train.copy()
train['y'] = y_train

# 将 y_val 和 X_val 合并
valid = X_val.copy()
valid['y'] = y_val

# 将 y_test 和 X_test 合并
test = X_test.copy()
test['y'] = y_test

# 保存到 CSV 文件
train.to_csv('C:\\Users\\syan\\Documents\\record\\phd\\2024\\molecular property prediction\\pin1\\train_pin1_1.csv', index=False)
valid.to_csv('C:\\Users\\syan\\Documents\\record\\phd\\2024\\molecular property prediction\\pin1\\valid_pin1_1.csv', index=False)
test.to_csv('C:\\Users\\syan\\Documents\\record\\phd\\2024\\molecular property prediction\\pin1\\test_pin1_1.csv', index=False)

print(len(X_train))
'''
zinc = pd.read_csv('dataset\\bioavailability.csv')
zinc = zinc.replace('\n', '', regex=True)

data = np.array(zinc['Drug'])
labels = np.array(zinc['Y'])


train_data, temp_data, train_labels, temp_labels = train_test_split(data, labels, test_size=0.2)
validation_data, test_data, validation_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5)

df_train = pd.DataFrame({'smiles':train_data, 'label':train_labels})
df_val = pd.DataFrame({'smiles':validation_data, 'label':validation_labels})
df_test = pd.DataFrame({'smiles':test_data, 'label': test_labels})

df_train.to_csv('train_bioavailability_1.csv', index=False)
df_val.to_csv('valid_bioavailability_1.csv', index=False)
df_test.to_csv('test_bioavailability_1.csv', index=False)
'''