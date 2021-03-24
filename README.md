# EMNIST Letters AI Lab

## L'elaborato
Si utilizzano **Random Forest Classifier** e **Decision Tree Classifier** di **Scikit-Learn** per classificare immagini di caratteri manoscritti del dataset EMNIST Letters (https://www.nist.gov/itl/products-and-services/emnist-dataset).

 Il file 'features_reduced.npz' è il risultato della Dimensionality Reduction. Le features originali sono omesse, scaricabili da NIST.
 
 Alle features sono associate le labels:
 - 'labels_original.npz' sono le labels originali di EMNIST Letters, convertite in matrici Numpy per semplicità di utilizzo
 - 'labels_corrected.npz' sono corrette come indicato sotto, e nella relazione
 

## Riprodurre i risultati

Installare librerie:

    pip install -U scikit-learn scikit-image matplotlib numpy

Per produrre le immagini analoghe a (Cohen et al. 2017):

    python decisiontree.py features_reduced.npz labels_original.npz
    python randomforest.py features_reduced.npz labels_original.npz 150

Per produrre le confusion matrix:

    python decisiontree.py features_reduced.npz labels_original.npz
    python decisiontree.py features_reduced.npz labels_corrected.npz

Tutti gli altri risultati delle tabelle sono riproducibili tramite l'utilizzo delle features e degli script nella cartella 'heuristics_utils':
### Labels 'Corrette'
Nella relazione si propongono delle labels alternative per risolvere il problema sistematico della confusione tra i caratteri i e L, prevedendo l'inserimento di una nuova classe 'linea verticale'. I dettagli si trovano nella relazione.

## Sintassi degli script

 Ogni script prende in input e restituisce in output file di matrici compresse Numpy '**.npz**'. Di seguito le sintassi per utilizzarli:
 
 Produci **Fourier Elliptic Descriptors** (2 mins):

    pip install pyefd
    python fourier.py features_original.npz output_file

Produci features **Profili delle Lettere** (15 mins):

    python profiles.py features_original.npz output_file

Produci features **Istogrammi di Proiezioni** (15 mins):

    python projection.py features_original.npz output_file

Produci features **Istogrammi di Grafi Orientati (hog)** (6 mins):

    python hog.py features_original.npz output_file

Produci **Labels Corrette** (4 mins):

    python edit.py labels_original.npz features_original.npz output_file

Applica **Principal Component Analysis**:

    python pca.py features.npz #_of_output_features output_file

Applica **Arrotondamento** delle features:

    python truncate.py features.npz #_of_decimal_digits output_file

**Convertire** le features originali in matrici numpy (.npz)

    python convert.py path_to_original_dataset

**Combina** features:

    python combine.py features1.npz features2.npz output_file

Avvia **Apprendimento**:

    python randomforest.py features.npz labels.npz #number_of_trees
    python decisiontree.py features.npz labels.npz decision_tree

## Come produrre i files

Come produrre il file '**features_reduced.npz**'?

    python convert.py path_to_dataset
    python fourier.py features_original.npz features_fourier
    python projection.py features_original.npz features_projection
    python pca.py features_projection.npz 7 features_projectionPCA
    python truncate.py features_projectionPCA.npz
    python combine.py features_projectionPCA.npz features_fourier.npz features_reduced

Come produrre il file '**labels_corrected.npz**'?

    python edit.py labels_original.npz features_original.npz labels_corrected



