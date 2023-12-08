import scipy
import pandas as pd
import numpy as np 
import skimage

cytoself_embeddings = np.load("C:\\Users\\amulya\\Downloads\\test_vqvec2_flat.npy")
labels = pd.read_csv("C:\\Users\\amulya\\Downloads\\test_label_nucenter.csv")
Z = np.load("C:\\Users\\amulya\\Downloads\\clustered_embeddings_cytoself_test.npy")

t = 1 # THRESHOLD TODO
clusters = scipy.cluster.hierarchy.fcluster(Z, t)
for c in clusters:
    # get pairwise embedding distance
    # TODO maybe not cosine
    self_similarities = scipy.spatial.distance.cosine(cytoself_embeddings[c], cytoself_embeddings[c])
    pair_most_diff = np.argmax(self_similarities)
    # are labels in same ordering as embeddings? assumes so
    most_diff_label = labels.loc[pair_most_diff[0]], labels.loc[pair_most_diff[1]]
    # can prob use this info to fetch and visualize images from online
    # OR AWS
    # ensg	name	loc_grade1	loc_grade2	loc_grade3	protein_id	FOV_id
    url0 = most_diff_label[0]
    image=skimage.io.imread()
