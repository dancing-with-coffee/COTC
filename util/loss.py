import numpy as np
import torch
from torch.nn import functional as F
EPS = 1e-9

###

def contrastive_loss(features_i, features_j, tau, cluster=None):
    batch_size = features_i.size(0)

    # similarity matrix between features
    features_cat = torch.cat([features_i, features_j])
    features_cat = F.normalize(features_cat, dim=1)
    similarity_matrix = torch.mm(features_cat, features_cat.T) / tau

    # positive pairs
    similarity_positive_ij = torch.diag(similarity_matrix, diagonal=batch_size)
    similarity_positive_ji = torch.diag(similarity_matrix, diagonal=-batch_size)
    similarity_positive_pairs = torch.cat([similarity_positive_ij, similarity_positive_ji]).reshape([-1, 1])

    if cluster != None:
        mask = (cluster.unsqueeze(1) == cluster.unsqueeze(0)).float()
        mask[torch.diag_embed(torch.diag(torch.ones_like(mask))).bool()] = 0
        similarity_positive_pairs = (similarity_positive_pairs + torch.sum(mask * similarity_matrix[: batch_size, : batch_size], dim=1, keepdim=True).repeat([2, 1])) / (1 + torch.sum(mask, dim=1, keepdim=True).repeat([2, 1]))

    # negative pairs
    mask = torch.ones_like(similarity_matrix)
    mask_ij = torch.diag(mask, diagonal=batch_size)
    mask_ji = torch.diag(mask, diagonal=-batch_size)
    mask = (mask - torch.diag_embed(torch.diag(mask)) - torch.diag_embed(mask_ij, offset=batch_size) - torch.diag_embed(mask_ji, offset=-batch_size)).bool()
    similarity_negative_pairs = similarity_matrix[mask].reshape([2 * batch_size, -1])

    # logits and labels
    logits = torch.cat([similarity_positive_pairs, similarity_negative_pairs], dim=1)
    labels = torch.zeros_like(logits[:, 0]).long()

    # final loss
    loss = F.cross_entropy(logits, labels)

    return loss

###

def ce_loss(logits, labels, masks=None):
    if masks == None:
        loss = F.cross_entropy(logits, labels)
    elif torch.sum(masks) > 0:
        loss = F.cross_entropy(logits[masks], labels[masks])
    else:
        loss = 0

    return loss

###

def consistency_loss(probabilities_i, probabilities_j, masks=None):
    if masks == None:
        loss = F.kl_div(torch.log(probabilities_i + EPS), probabilities_j + EPS, reduction="batchmean")
    elif torch.sum(masks) > 0:
        loss = F.kl_div(torch.log(probabilities_i[masks] + EPS), probabilities_j[masks] + EPS, reduction="batchmean")
    else:
        loss = 0

    return loss

###

def vae_loss(sign_x, log_x_hat, mu, sigma):
    reconstruction = reconstruction_loss(sign_x, log_x_hat)
    divergence = vae_divergence_loss(mu, sigma)

    loss = reconstruction + 0.1 * divergence

    return loss

###

def vade_loss(sign_x, log_x_hat, gamma, mu, sigma, cluster_pi, cluster_mu, cluster_sigma):
    reconstruction = reconstruction_loss(sign_x, log_x_hat)
    divergence = vade_divergence_loss(gamma, mu, sigma, cluster_pi, cluster_mu, cluster_sigma)

    loss = reconstruction + divergence

    return loss

###

def sample_loss(sign_x, log_x_hat, z, sigma, cluster_pi, cluster_mu, cluster_sigma):
    reconstruction = reconstruction_loss(sign_x, log_x_hat)
    divergence = sample_divergence_loss(z, sigma, cluster_pi, cluster_mu, cluster_sigma)

    loss = reconstruction + divergence

    return loss

###

def reconstruction_loss(sign_x, log_x_hat):
    loss = -torch.mean(torch.sum(sign_x * log_x_hat, dim=1))

    return loss

###

def vae_divergence_loss(mu, sigma):
    loss = 1 / 2 * torch.mean(torch.sum(mu ** 2 + sigma ** 2 - torch.log(sigma ** 2 + EPS), dim=1))

    return loss

###

def vade_divergence_loss(gamma, mu, sigma, cluster_pi, cluster_mu, cluster_sigma):
    loss_z = -1 / 2 * torch.mean(torch.sum(torch.log(sigma ** 2 + EPS), dim=1))
    loss_c = torch.mean(torch.sum(F.kl_div(torch.log(cluster_pi + EPS), gamma + EPS, reduction="none"), dim=1))

    mu = mu.unsqueeze(1)
    sigma = sigma.unsqueeze(1)
    cluster_mu = cluster_mu.unsqueeze(0)
    cluster_sigma = cluster_sigma.unsqueeze(0)
    temp = torch.sum(((mu - cluster_mu) ** 2 + sigma ** 2) / (cluster_sigma ** 2 + EPS) + torch.log(cluster_sigma ** 2 + EPS), dim=-1)
    loss_zc = 1 / 2 * torch.mean(torch.sum(gamma * temp, dim=1))

    loss = loss_z + loss_c + loss_zc

    return loss

###

def sample_divergence_loss(z, sigma, cluster_pi, cluster_mu, cluster_sigma):
    loss_0 = -1 / 2 * torch.mean(torch.sum(torch.log(sigma ** 2 + EPS), dim=1))

    z = z.unsqueeze(1)
    cluster_mu = cluster_mu.unsqueeze(0)
    cluster_sigma = cluster_sigma.unsqueeze(0)
    temp = torch.sum(((z - cluster_mu) / (cluster_sigma + EPS)) ** 2 + torch.log(cluster_sigma ** 2 + EPS) + np.log(2 * np.pi), dim=-1)
    loss_1 = -torch.mean(torch.log(torch.sum(cluster_pi * torch.exp(-1 / 2 * temp), dim=1) + EPS))

    loss = loss_0 + loss_1

    return loss

###
