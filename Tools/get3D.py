import requests
import pandas as pd

fail = []
def is_nan(value):
    try:
        float(value)
        return False
    except ValueError:
        return True

def get3D(uniprot_id, file='cif'):
    #file: 'cif' or 'pdb'
    cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.{file}"
    r = requests.get(cif_url, timeout=60)
    if r.status_code == 200:
        cif_filename = f"{uniprot_id}.{file}"
        with open(cif_filename, "wb") as f:
            f.write(r.content)
        #print(f"CIF file downloaded: {cif_filename}")
    else:
        fail.append(uniprot_id)
        #print(f"Failed to download CIF file for {uniprot_id}, status code: {r.status_code}")
        return

uniprot_id = pd.read_csv('kiba_uniprot.csv')['uniprotid']
for i in range(len(uniprot_id)):
    uniprot_id_value = uniprot_id[i]

    # Ensure the value is a string before splitting
    if isinstance(uniprot_id_value, str):
        print(uniprot_id_value)
        uniprot_id_part = uniprot_id_value.split()[0]
        get3D(uniprot_id_part)

fail = pd.DataFrame(fail)
fail.to_csv('fail.csv', index=False)
