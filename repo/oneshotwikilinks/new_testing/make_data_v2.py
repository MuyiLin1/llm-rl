"""
make_data_v2.py

Rebuilds mydataset.2000.v2.pickle.gz using all-mpnet-base-v2 instead of
bert-base-nli-mean-tokens. Both entity embeddings and context embeddings
are re-embedded in the new space.

Usage (run from the directory containing entityfreq.gz and shuffled_dedup_entities.tsv):
    python make_data_v2.py

Output: mydataset.2000.v2.pickle.gz
"""

import torch
import re
import gzip
import pickle
from collections import defaultdict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ST_MODEL = 'all-mpnet-base-v2'  # 768-dim, much better than bert-base-nli-mean-tokens


def categoryCount():
    counts = {}
    with gzip.open('entityfreq.gz', 'rt') as f:
        for line in f:
            try:
                freq, entity = line.strip().split()
            except:
                continue
            counts[entity] = int(freq)
    return counts


def getCategories(threshold, model):
    for entity, freq in categoryCount().items():
        if freq >= threshold:
            niceentity = re.sub(r'_', r' ', entity)
            embedcat = model.encode([niceentity])[0]
            yield entity, embedcat


def makeData(threshold, categories, model):
    catcount = defaultdict(int)

    with open('shuffled_dedup_entities.tsv') as f:
        batchline, batchencode, batchentity = [], [], []
        for line in f:
            try:
                entity, pre, mention, post = line.strip().split('\t')
            except:
                continue

            if entity in categories and catcount[entity] < threshold:
                catcount[entity] += 1
                batchline.append(line)
                batchencode.append(pre)
                batchencode.append(post)
                batchentity.append(entity)

                if len(batchline) == 5:
                    embed = model.encode(batchencode)

                    for n, (line, entity) in enumerate(zip(batchline, batchentity)):
                        embedpre, embedpost = embed[2*n], embed[2*n+1]
                        entityord, entityvec = categories[entity]
                        yield {'line': line,
                               'entityord': entityord,
                               'entityvec': entityvec,
                               'pre': embedpre,
                               'post': embedpost}

                    batchline, batchencode, batchentity = [], [], []

        if len(batchline):
            embed = model.encode(batchencode)

            for n, (line, entity) in enumerate(zip(batchline, batchentity)):
                embedpre, embedpost = embed[2*n], embed[2*n+1]
                entityord, entityvec = categories[entity]
                yield {'line': line,
                       'entityord': entityord,
                       'entityvec': entityvec,
                       'pre': embedpre,
                       'post': embedpost}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold, model):
        self.labelfeats = {k: (n, v) for n, (k, v) in enumerate(getCategories(threshold, model))}
        Xs = []
        ys = []
        pre_texts = []
        post_texts = []
        text_entities = []
        for n, what in tqdm(enumerate(makeData(threshold, self.labelfeats, model)),
                            desc="Building dataset"):
            pre = torch.tensor(what['pre'])
            post = torch.tensor(what['post'])
            Xs.append(torch.cat((pre, post)).unsqueeze(0))
            ys.append(what['entityord'])

            text_entity, text_pre, text_mention, text_post = what['line'].strip().split('\t')
            text_niceentity = re.sub(r'_', r' ', text_entity)

            pre_texts.append(text_pre)
            post_texts.append(text_post)
            text_entities.append(text_niceentity)

        self.Xs = torch.cat(Xs, dim=0)
        self.ys = torch.LongTensor(ys)
        self.pre_texts = pre_texts
        self.post_texts = post_texts
        self.text_entities = text_entities

    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], self.pre_texts[index], self.post_texts[index], self.text_entities[index]


if __name__ == '__main__':
    print(f"Using SentenceTransformer: {ST_MODEL}")
    model = SentenceTransformer(ST_MODEL)

    threshold = 2000
    numclasses = len([e for e, f in categoryCount().items() if f >= threshold])
    print(f"Threshold: {threshold}, Classes: {numclasses}, Samples: {threshold * numclasses}")

    print("Building dataset (this will take a while)...")
    dataset = MyDataset(threshold, model)
    print(f"Dataset size: {len(dataset)} samples, Xs shape: {dataset.Xs.shape}")

    fname = f'mydataset.{threshold}.v2.pickle.gz'
    with gzip.open(fname, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved to {fname}")
