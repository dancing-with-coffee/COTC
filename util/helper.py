from timeit import default_timer
from datetime import timedelta
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import seaborn as sns
EPS = 1e-9

###

"""
easy timer method
review passed
"""
def timer(start=None):
    if start == None:
        return default_timer()
    else:
        return str(timedelta(seconds=round(default_timer() - start)))

###

"""
abstract implementation of train loop
review passed
"""
def train_loop(loader, convert, forward, optimizer, pre_action="", post_action="", *args):
    ret = [{}, 0] # [output_sum, num_steps]

    for iteration, batch in enumerate(loader):
        exec(pre_action)

        input = convert(batch)
        output = forward(*input)

        for key, value in output.items():
            if key in ret[0]:
                ret[0][key] = ret[0][key] + value
            else:
                ret[0][key] = value

        ret[1] = ret[1] + 1

        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()

        exec(post_action)

    return ret

###

"""
abstract implementation of test loop
review passed
"""
def test_loop(loader, convert, forward):
    ret = [] # desired outcome

    with torch.no_grad():
        for iteration, batch in enumerate(loader):
            input = convert(batch)
            output = forward(*input)

            ret.append(output)

    ret = torch.cat(ret)

    return ret

###

"""
searching neighbor method knn
x required to be torch tensor
review passed
"""
def knn(x, k):
    x = F.normalize(x, dim=1)

    indexes = []
    chunk_size = 1000

    for i in range(0, x.size(0), chunk_size):
        chunk_x = x[i: i + chunk_size]
        chunk_scores = torch.mm(chunk_x, x.T)
        _, chunk_indexes = torch.topk(chunk_scores, k=k + 1, dim=1)

        indexes.append(chunk_indexes[:, 1:])

    indexes = torch.cat(indexes)

    return indexes

###

"""
clustering method kmeans
x required to be torch tensor
review passed
"""
def kmeans(x, n_clusters, max_iter=100, n_init=1, init=None, pseudo=None, metric=None):
    n = x.size(0)

    best_score = None
    best_mu = None
    best_y = None
    n_init = 1 if init != None else n_init

    for i in range(n_init):
        if init != None:
            mu = init
        elif pseudo != None:
            mu = torch.cat([x[pseudo == j][torch.randperm(x[pseudo == j].size(0))[0]].reshape([1, -1]) for j in range(n_clusters)])
        else:
            mu = kmeans_plus_plus(x, n_clusters=n_clusters)

        for j in range(max_iter):
            y = []
            chunk_size = 1000

            for k in range(0, x.size(0), chunk_size):
                chunk_x = x[k: k + chunk_size]
                chunk_y = torch.argmin(torch.sum((chunk_x.unsqueeze(1) - mu.unsqueeze(0)) ** 2, dim=-1), dim=1)

                y.append(chunk_y)

            y = torch.cat(y)

            mu = torch.cat([torch.mean(x[y == k], dim=0, keepdim=True) for k in range(n_clusters)])

            ind_nan = torch.any(torch.isnan(mu), dim=1)
            num_nan = torch.sum(ind_nan)
            mu[ind_nan] = x[torch.randperm(n)[: num_nan]]

            ind_inf = torch.any(torch.isinf(mu), dim=1)
            num_inf = torch.sum(ind_inf)
            mu[ind_inf] = x[torch.randperm(n)[: num_inf]]

        if n_init != 1:
            if metric == "ss":
                score = silhouette_score(x, y) # positive, the higher, the better
            elif metric == "dbs":
                score = -davies_bouldin_score(x, y) # negative, the lower, the better
            elif metric == "chs":
                score = calinski_harabasz_score(x, y) # positive, the higher, the better
            else:
                score = -inertia_score(x, y) # negative, the lower, the better

            if best_score == None or score > best_score:
                best_score = score
                best_mu = mu
                best_y = y
        else:
            best_mu = mu
            best_y = y

    return best_mu, best_y

###

"""
clustering method kmeans implemented using sklearn
x required to be numpy array
review passed
"""
def kmeans_sklearn(x, n_clusters, max_iter=100, n_init=1):
    means = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)

    y = means.fit_predict(x)
    mu = means.cluster_centers_

    return mu, y

###

"""
clustering method gmm implemented using sklearn
x required to be numpy array
review passed
"""
def gmm_sklearn(x, n_clusters, max_iter=100, n_init=1):
    mixture = GaussianMixture(n_components=n_clusters, covariance_type="diag", max_iter=max_iter, n_init=n_init)

    y = mixture.fit_predict(x)
    mu = mixture.means_

    return mu, y

###

"""
clustering method ahc implemented using sklearn
x required to be numpy array
review passed
"""
def ahc_sklearn(x, n_clusters, max_iter=100, n_init=1):
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="complete")

    y = clustering.fit_predict(x)
    mu = np.concatenate([np.mean(x[y == i], axis=0, keepdims=True) for i in range(n_clusters)])

    return mu, y

###

"""
visualization method tsne implemented using sklearn
x and y required to be numpy array
review passed
"""
def tsne_sklearn(x, y, path, perplexity=8, s=8):
    tsne = TSNE(perplexity=perplexity)

    z = tsne.fit_transform(x)
    z_max = np.max(z, axis=0)
    z_min = np.min(z, axis=0)
    z = (z - z_min) / (z_max - z_min)

    plt.figure(figsize=[8, 8])
    plt.scatter(z[:, 0], z[:, 1], s=s, c=y, cmap="Spectral")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

###

"""
visualization method heatmap implemented using seaborn
p and y required to be numpy array
review passed
"""
def heatmap_seaborn(p, y, path):
    d = max(np.max(p), np.max(y)) + 1
    w = np.zeros([d, d])

    for i in range(p.shape[0]):
        w[p[i], y[i]] = w[p[i], y[i]] + 1

    ind_row, ind_col = linear_sum_assignment(np.max(w) - w)

    ind_dic = dict(zip(ind_col, ind_row))
    ind_map = [ind_dic[i] for i in range(len(ind_dic))]

    plt.figure(figsize=[d, d])
    sns.heatmap(w[ind_map], annot=True, fmt="g", square=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()

###

"""
optimal transport solver method sinkhorn knopp
M and a and b required to be torch tensor
review passed
"""
def sinkhorn_knopp(M, a, b, epsilon, variable=False, regularization=None, threshold=None):
    u = torch.ones_like(a) / a.size(0)
    v = torch.ones_like(b) / b.size(0)

    K = torch.exp(-M / epsilon)

    for i in range(1000):
        u_ = u
        v_ = v

        # scale vector u v to avoid overflow
        u_max = torch.log(torch.max(u))
        u_min = torch.log(torch.min(u))
        u_avg = (u_max + u_min) / 2

        v_max = torch.log(torch.max(v))
        v_min = torch.log(torch.min(v))
        v_avg = (v_max + v_min) / 2

        gap = torch.abs(u_avg - v_avg)

        if u_avg > v_avg:
            u = u / torch.exp(gap / 2)
            v = v * torch.exp(gap / 2)
        else:
            u = u * torch.exp(gap / 2)
            v = v / torch.exp(gap / 2)

        u = a / torch.mm(K, v)
        v = b / torch.mm(K.T, u)

        if torch.any(torch.isinf(u)) or torch.any(torch.isnan(u)) or torch.any(torch.isinf(v)) or torch.any(torch.isnan(v)):
            u = u_
            v = v_

            break

        if variable:
            g = epsilon * torch.log(v)

            # search point p to avoid divergence
            g_0 = g
            g_0[g_0 == 0] = EPS
            sqrt_delta = torch.sqrt(g_0 ** 2 + 4 * regularization ** 2)
            f_0 = torch.sum(1 / 2 + regularization / g_0 - sqrt_delta / (2 * g_0)) - 1

            flag = False

            for coefficient in [1, -1]:
                p = coefficient * torch.arange(0, 10, 0.001).reshape([1, -1]).to(g.device)

                g_p = g - p
                g_p[g_p == 0] = EPS
                sqrt_delta = torch.sqrt(g_p ** 2 + 4 * regularization ** 2)
                f_p = torch.sum(1 / 2 + regularization / g_p - sqrt_delta / (2 * g_p), dim=0) - 1

                if f_0 > 0:
                    indexes = torch.where(f_p < 0)[0]
                else:
                    indexes = torch.where(f_p > 0)[0]

                if indexes.size(0) != 0:
                    flag = True

                    h = p[0, indexes[0]]

                    break

            if not flag:
                h = 1

            for j in range(10):
                g_h = g - h
                g_h[g_h == 0] = EPS
                sqrt_delta = torch.sqrt(g_h ** 2 + 4 * regularization ** 2)
                f_h = torch.sum(1 / 2 + regularization / g_h - sqrt_delta / (2 * g_h)) - 1
                f_h_ = torch.sum(regularization / g_h ** 2 + 1 / (2 * sqrt_delta) - sqrt_delta / (2 * g_h ** 2))

                e = torch.abs(f_h)
                s = torch.abs(f_h / f_h_)
                h = h - f_h / f_h_

                if e < threshold or s < threshold:
                    g_h = g - h
                    g_h[g_h == 0] = EPS
                    b = 1 / 2 + regularization / g_h - sqrt_delta / (2 * g_h)

                    break

        if torch.norm(u - u_) < EPS or torch.norm(v - v_) < EPS:
            break

    Pi = torch.einsum("i,ij,j->ij", u.reshape(-1), K, v.reshape(-1))

    if variable:
        return Pi, b
    else:
        return Pi

###

"""
clustering initialization method kmeans plus plus
x required to be torch tensor
review passed
"""
def kmeans_plus_plus(x, n_clusters):
    mu = [x[torch.randperm(x.size(0))[0]].reshape([1, -1])]
    records = torch.sum((x.unsqueeze(1) - mu[0].unsqueeze(0)) ** 2, dim=-1).reshape(-1)

    for i in range(1, n_clusters):
        probabilities = records / torch.sum(records)
        candidates = x[torch.distributions.categorical.Categorical(probabilities).sample([int(np.log(n_clusters)) + 2])]
        distances = torch.sum((x.unsqueeze(1) - candidates.unsqueeze(0)) ** 2, dim=-1).T
        temp = torch.argmin(torch.sum(torch.where(distances < records, distances, records), dim=1))

        mu.append(candidates[temp].reshape([1, -1]))
        records = distances[temp]

    mu = torch.cat(mu)

    return mu

###

"""
clustering metric method silhouette score
x and p required to be torch tensor
review passed
"""
def silhouette_score(x, p):
    n = x.size(0)
    k = torch.max(p) + 1

    distances = []
    chunk_size = 1000

    for i in range(0, n, chunk_size):
        chunk_distances = []

        for j in range(0, n, chunk_size):
            chunk_x_0 = x[i: i + chunk_size]
            chunk_x_1 = x[j: j + chunk_size]
            chunk_chunk_distances = torch.sqrt(torch.sum((chunk_x_0.unsqueeze(1) - chunk_x_1.unsqueeze(0)) ** 2, dim=-1))

            chunk_distances.append(chunk_chunk_distances)

        chunk_distances = torch.cat(chunk_distances, dim=1)

        distances.append(chunk_distances)

    distances = torch.cat(distances)

    all_clusters = []

    for i in range(k):
        all_clusters.append(torch.mean(distances[:, p == i], dim=1, keepdim=True))

    all_clusters = torch.cat(all_clusters, dim=1)

    for i in range(k):
        indexes = p == i
        temp = torch.sum(indexes)

        if temp != 1:
            all_clusters[indexes, i] = temp / (temp - 1) * all_clusters[indexes, i]

    intra_clusters = torch.zeros_like(all_clusters[:, 0])
    inter_clusters = torch.zeros_like(all_clusters[:, 0])

    for i in range(k):
        indexes = p == i

        intra_clusters[indexes] = all_clusters[indexes, i]
        inter_clusters[indexes] = torch.min(all_clusters[indexes][:, [j for j in range(k) if j != i]], dim=1)[0]

    score = torch.zeros_like(all_clusters[:, 0])

    indexes = intra_clusters > inter_clusters
    score[indexes] = inter_clusters[indexes] / intra_clusters[indexes] - 1
    indexes = intra_clusters < inter_clusters
    score[indexes] = 1 - intra_clusters[indexes] / inter_clusters[indexes]

    score = torch.mean(score)

    return score

###

"""
clustering metric method davies bouldin score
x and p required to be torch tensor
review passed
"""
def davies_bouldin_score(x, p):
    k = torch.max(p) + 1
    mu = torch.cat([torch.mean(x[p == i], dim=0, keepdim=True) for i in range(k)])

    intra_clusters = []

    for i in range(k):
        intra_clusters.append(torch.mean(torch.sqrt(torch.sum((x[p == i] - mu[i]) ** 2, dim=1))).reshape(-1))

    intra_clusters = torch.cat(intra_clusters)

    intra_clusters = intra_clusters.unsqueeze(1) + intra_clusters.unsqueeze(0)
    inter_clusters = torch.sqrt(torch.sum((mu.unsqueeze(1) - mu.unsqueeze(0)) ** 2, dim=-1))

    mask = torch.ones_like(intra_clusters)
    mask = (mask - torch.diag_embed(torch.diag(mask))).bool()

    intra_clusters = intra_clusters[mask].reshape([k, -1])
    inter_clusters = inter_clusters[mask].reshape([k, -1])

    score = torch.zeros_like(intra_clusters)

    indexes = intra_clusters != 0
    score[indexes] = intra_clusters[indexes] / inter_clusters[indexes]

    score = torch.mean(torch.max(score, dim=1)[0])

    return score

###

"""
clustering metric method calinski harabasz score
x and p required to be torch tensor
review passed
"""
def calinski_harabasz_score(x, p):
    n = x.size(0)
    k = torch.max(p) + 1
    mu = torch.cat([torch.mean(x[p == i], dim=0, keepdim=True) for i in range(k)])
    M = torch.mean(x, dim=0)

    intra_clusters = 0
    inter_clusters = 0

    for i in range(k):
        indexes = p == i

        intra_clusters = intra_clusters + torch.sum((x[indexes] - mu[i]) ** 2)
        inter_clusters = inter_clusters + torch.sum(indexes) * torch.sum((mu[i] - M) ** 2)

    score = (n - k) / (k - 1) * inter_clusters / intra_clusters

    return score

###

"""
clustering metric method inertia score
x and p required to be torch tensor
review passed
"""
def inertia_score(x, p):
    k = torch.max(p) + 1
    mu = torch.cat([torch.mean(x[p == i], dim=0, keepdim=True) for i in range(k)])

    score = 0

    for i in range(k):
        score = score + torch.sum((x[p == i] - mu[i]) ** 2)

    return score

###
