# Molecular Property Prediction Datasets
- **Datasets**
  1. MoleculeNet: [dataset](https://moleculenet.org/)  
  2. BioADME: [dataset](https://github.com/molecularinformatics/Computational-ADME)
  3. CYP: [dataset](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00628)
  4. Caco2:
     [paper](https://pubmed.ncbi.nlm.nih.gov/27018227/)
     [dataset](https://github.com/Duke-W91/Caco2_prediction/tree/main)
  5. MoleculeACE:
     [paper](https://pubs.acs.org/doi/epdf/10.1021/acs.jcim.2c01073?ref=article_openPDF)
     [dataset](https://github.com/molML/MoleculeACE/tree/main/MoleculeACE/Data/benchmark_data)
- **Problem with BBBP dataset**: 12 SMILES strings in the BBBP dataset fail when parsing with rdkit, all are removed in our splits.   
  https://github.com/deepchem/deepchem/issues/2336  
  These are the smiles strings:  
  59 O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3  
  61 c1(nc(NC(N)=[NH2])sc1)CSCCNC(=[NH]C#N)NC  
  391 Cc1nc(sc1)[NH]=C(\N)N  
  614 s1cc(CSCCN\C(NC)=[NH]\C#N)nc1[NH]=C(\N)N  
  642 c1c(c(ncc1)CSCCN\C(=[NH]\C#N)NCC)Br  
  645 n1c(csc1[NH]=C(\N)N)c1ccccc1  
  646 n1c(csc1[NH]=C(\N)N)c1cccc(c1)N  
  647 n1c(csc1[NH]=C(\N)N)c1cccc(c1)NC(C)=O  
  648 n1c(csc1[NH]=C(\N)N)c1cccc(c1)N\C(NC)=[NH]\C#N  
  649 s1cc(nc1[NH]=C(\N)N)C  
  685 c1(cc(N\C(=[NH]\c2cccc(c2)CC)C)ccc1)CC  
  1998 [C@@h]3(C1=CC=C(Cl)C=C1)[C@H]2CCC@@HC34C...  
