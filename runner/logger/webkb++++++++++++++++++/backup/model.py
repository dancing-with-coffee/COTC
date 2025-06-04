from util.packet import *

###

from collections import Counter
from transformers import AutoConfig, AutoModel
from util.loss import contrastive_loss, ce_loss, consistency_loss, vae_loss, vade_loss
EPS = 1e-9

###

class Model(TemplateModel):
    def __init__(self, config, logger):
        super().__init__(config, logger)

    def define(self):
        # sbert backbone
        config = AutoConfig.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
        config.attention_dropout = self.config.dropout
        config.dropout = self.config.dropout

        self.sbert = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", config=config)

        # projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.sbert.config.hidden_size, 768),
            nn.ReLU(),
            nn.Linear(768, self.config.feature_length)
        )

        # clustering layer
        self.clustering = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.sbert.config.hidden_size, 768),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, self.config.num_clusters)
        )

        # gaussian component
        self.cluster_pi = nn.Parameter(torch.ones(self.config.num_clusters) / self.config.num_clusters)
        self.cluster_mu = nn.Parameter(torch.zeros([self.config.num_clusters, self.config.feature_length]))
        self.cluster_sigma = nn.Parameter(torch.ones([self.config.num_clusters, self.config.feature_length]))

        # encoder mu of z
        self.encoder_mu = nn.Sequential(
            nn.Linear(self.config.max_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.config.feature_length)
        )

        # encoder sigma of z
        self.encoder_sigma = nn.Sequential(
            nn.Linear(self.config.max_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.config.feature_length)
        )

        # decoder of x
        self.decoder = nn.Sequential(
            nn.Linear(self.config.feature_length, self.config.max_features)
        )

    def forward(self, texts, texts_0, texts_1, tfidfs, texts_n, tfidfs_n, labels, masks):
        attention_mask = texts["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(self.sbert(**texts)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        attention_mask = texts_0["attention_mask"].unsqueeze(-1)
        embeddings_0 = torch.sum(self.sbert(**texts_0)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        attention_mask = texts_1["attention_mask"].unsqueeze(-1)
        embeddings_1 = torch.sum(self.sbert(**texts_1)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        attention_mask = texts_n["attention_mask"].unsqueeze(-1)
        embeddings_n = torch.sum(self.sbert(**texts_n)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        features = F.normalize(self.projection(embeddings), dim=1)
        features_0 = F.normalize(self.projection(embeddings_0), dim=1)
        features_1 = F.normalize(self.projection(embeddings_1), dim=1)
        features_n = F.normalize(self.projection(embeddings_n), dim=1)

        logits = self.clustering(embeddings)
        logits_0 = self.clustering(embeddings_0)
        logits_1 = self.clustering(embeddings_1)
        logits_n = self.clustering(embeddings_n)

        probabilities = F.softmax(logits, dim=1)
        probabilities_0 = F.softmax(logits_0, dim=1)
        probabilities_1 = F.softmax(logits_1, dim=1)
        probabilities_n = F.softmax(logits_n, dim=1)

        gamma_b = F.gumbel_softmax((logits_0 + logits_1 + logits_n) / 3, tau=self.config.tau)

        mu = torch.tanh(self.encoder_mu(tfidfs) / 3) * 3
        mu_n = torch.tanh(self.encoder_mu(tfidfs_n) / 3) * 3

        sigma = torch.exp(self.encoder_sigma(tfidfs) / 2)
        sigma_n = torch.exp(self.encoder_sigma(tfidfs_n) / 2)

        z = mu + torch.randn_like(sigma) * sigma
        z_n = mu + torch.randn_like(sigma_n) * sigma_n

        sign_x = torch.sign(tfidfs)
        log_x_hat = F.log_softmax(self.decoder(z), dim=1)

        z = z.unsqueeze(1)
        cluster_mu = self.cluster_mu.unsqueeze(0)
        cluster_sigma = self.cluster_sigma.unsqueeze(0)
        temp = -1 / 2 * torch.sum(((z - cluster_mu) / (cluster_sigma + EPS)) ** 2 + torch.log(cluster_sigma ** 2 + EPS), dim=-1) + torch.log(self.cluster_pi + EPS)
        gamma_t = F.softmax(temp, dim=1)

        cluster_b = torch.argmax(gamma_b, dim=1)
        cluster_t = torch.argmax(gamma_t, dim=1)

        gamma = self.config.zeta * gamma_t + (1 - self.config.zeta) * gamma_b

        contrastive_loss_01 = contrastive_loss(features_0, features_1, tau=0.5)
        contrastive_loss_0n = contrastive_loss(features_0, features_n, tau=0.5, cluster=cluster_t)
        contrastive_loss_1n = contrastive_loss(features_1, features_n, tau=0.5, cluster=cluster_t)

        contrastive_loss_ = contrastive_loss_01 + contrastive_loss_0n + contrastive_loss_1n

        ce_loss_0 = ce_loss(logits_0, labels, masks)
        ce_loss_1 = ce_loss(logits_1, labels, masks)
        ce_loss_n = ce_loss(logits_n, labels, masks)

        ce_loss_ = ce_loss_0 + ce_loss_1 + ce_loss_n

        consistency_loss_0 = consistency_loss(probabilities_0, probabilities, masks)
        consistency_loss_1 = consistency_loss(probabilities_1, probabilities, masks)
        consistency_loss_n = consistency_loss(probabilities_n, probabilities, masks)

        consistency_loss_ = consistency_loss_0 + consistency_loss_1 + consistency_loss_n

        loss_b = contrastive_loss_ + consistency_loss_ + ce_loss_
        loss_t = vade_loss(sign_x, log_x_hat, gamma, mu, sigma, self.cluster_pi, self.cluster_mu, self.cluster_sigma) + contrastive_loss(z.squeeze(1), z_n, tau=0.5, cluster=cluster_b)

        loss = self.config.alpha * loss_b + self.config.beta * loss_t

        return {"loss": loss}

    def define_mu(self, mu):
        self.cluster_mu = nn.Parameter(mu)

    def forward_pre(self, tfidfs, tfidfs_n):
        mu = torch.tanh(self.encoder_mu(tfidfs) / 3) * 3
        sigma = torch.exp(self.encoder_sigma(tfidfs) / 2)
        z = mu + torch.randn_like(sigma) * sigma

        sign_x = torch.sign(tfidfs)
        log_x_hat = F.log_softmax(self.decoder(z), dim=1)

        mu_n = torch.tanh(self.encoder_mu(tfidfs_n) / 3) * 3
        sigma_n = torch.exp(self.encoder_sigma(tfidfs_n) / 2)
        z_n = mu_n + torch.randn_like(sigma_n) * sigma_n

        loss = vae_loss(sign_x, log_x_hat, mu, sigma) + self.config.eta * contrastive_loss(z, z_n, tau=0.5) # special strategy for pre-training

        return {"loss": loss}

    def embed(self, texts):
        attention_mask = texts["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(self.sbert(**texts)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)

        return embeddings

    def project(self, texts):
        attention_mask = texts["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(self.sbert(**texts)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        features = F.normalize(self.projection(embeddings), dim=1)

        return features

    def score_b(self, texts):
        attention_mask = texts["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(self.sbert(**texts)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        logits = self.clustering(embeddings)
        probabilities = F.softmax(logits, dim=1)

        return probabilities

    def cluster_b(self, texts):
        attention_mask = texts["attention_mask"].unsqueeze(-1)
        embeddings = torch.sum(self.sbert(**texts)[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        logits = self.clustering(embeddings)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions

    def encode(self, tfidfs):
        z = torch.tanh(self.encoder_mu(tfidfs) / 3) * 3

        return z

    def score_t(self, tfidfs):
        z = torch.tanh(self.encoder_mu(tfidfs) / 3) * 3

        z = z.unsqueeze(1)
        cluster_mu = self.cluster_mu.unsqueeze(0)
        cluster_sigma = self.cluster_sigma.unsqueeze(0)
        temp = -1 / 2 * torch.sum(((z - cluster_mu) / (cluster_sigma + EPS)) ** 2 + torch.log(cluster_sigma ** 2 + EPS), dim=-1) + torch.log(self.cluster_pi + EPS)
        gamma = F.softmax(temp, dim=1)

        return gamma

    def cluster_t(self, tfidfs):
        z = torch.tanh(self.encoder_mu(tfidfs) / 3) * 3

        z = z.unsqueeze(1)
        cluster_mu = self.cluster_mu.unsqueeze(0)
        cluster_sigma = self.cluster_sigma.unsqueeze(0)
        temp = -1 / 2 * torch.sum(((z - cluster_mu) / (cluster_sigma + EPS)) ** 2 + torch.log(cluster_sigma ** 2 + EPS), dim=-1) + torch.log(self.cluster_pi + EPS)
        gamma = F.softmax(temp, dim=1)

        predictions = torch.argmax(gamma, dim=1)

        return predictions

###

def drop(model, dropout):
    for _, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout

###
