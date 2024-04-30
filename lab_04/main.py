import pandas as pd
import xml.etree.ElementTree as ET
import base64
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import re
from fuzzywuzzy import process, fuzz
import numpy as np

LEN_SHINGLE = 5
MATCH_THRESHOLD = 70
CLUSTER_THRESHOLD = 0.2

def find_shingles(text):
    items = text.split()
    shingles = []
    for i in range(len(items) - LEN_SHINGLE + 1):
        shingle = items[i:i+LEN_SHINGLE]
        print(' '.join(shingle))
        shingles.append(' '.join(shingle))

    return shingles


def compare_shingles(sh1, sh2):
    matches_cnt = 0
    for shingle in sh1:
        match = process.extractOne(shingle, sh2)
        if match[1] > MATCH_THRESHOLD:
            matches_cnt += 1

    return matches_cnt / ((len(sh1) + len(sh2))/2)


data = np.genfromtxt('filtered.csv', delimiter=',', encoding='utf8', dtype=None)
n = len(data)

shingles = []
for i in range(n):
    shingles.append(find_shingles(data[i]))

matches = []
clusters = []

for i in range(n):
    print(f'Text {i}:')
    matches.append([])

    for j in range(n):
        if i != j:
            metric = compare_shingles(shingles[i], shingles[j])
            print(f'--- text {j}: threshold = {metric}, match = ', metric > CLUSTER_THRESHOLD)
            if metric > CLUSTER_THRESHOLD:
                matches[i].append(j)
                already_append = False
                for cluster in clusters:
                    if i in cluster or j in cluster:
                        cluster.add(i)
                        cluster.add(j)
                        already_append = True

                if not already_append:
                    clusters.append({i, j})

    print('\n----------------------------------------------------------\n')

print(f'Clusters: {clusters})
