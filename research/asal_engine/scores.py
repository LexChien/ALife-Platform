import numpy as np

def cosine(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

def supervised_target_score(img_embs, txt_emb):
    return cosine(img_embs[-1], txt_emb)

def openended_score(img_embs):
    sims = []
    for t in range(1, len(img_embs)):
        cur = img_embs[t]
        past = img_embs[:t]
        nn = max(cosine(cur, p) for p in past)
        sims.append(nn)
    return -float(np.mean(sims) if sims else 0.0)

def illumination_diversity(img_embs_list):
    if len(img_embs_list) <= 1:
        return 0.0
    arr = np.stack(img_embs_list)
    arr = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    sims = arr @ arr.T
    np.fill_diagonal(sims, -1.0)
    return -float(sims.max(axis=1).mean())
