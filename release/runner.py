from util.packet import *

###

from util.helper import (
    timer,
    train_loop,
    test_loop,
    knn,
    kmeans,
    ahc_sklearn,
    sinkhorn_knopp,
)
from util.eval import neighbor_precision, clustering_acc, clustering_nmi

###


class Runner(TemplateRunner):
    def __init__(self, config, logger, data, model):
        super().__init__(config, logger, data=data, model=model)

        _, self.train_dataset, _ = self.data.load_datasets()
        self.pre_loader, self.train_loader, self.test_loader = self.data.get_loaders()

        self.model.define()
        self.model.to(self.device)

        if self.config.pre:
            self.optimizer = optim.Adam(
                [
                    {
                        "params": self.model.encoder_mu.parameters(),
                        "lr": self.config.lr_pre,
                    },
                    {
                        "params": self.model.encoder_sigma.parameters(),
                        "lr": self.config.lr_pre,
                    },
                    {
                        "params": self.model.decoder.parameters(),
                        "lr": self.config.lr_pre,
                    },
                ]
            )
        else:
            self.model.load_state_dict(
                torch.load("model/" + self.config.model_name + "-pre.pt")
            )

            self.optimizer = optim.Adam(
                [
                    {
                        "params": self.model.sbert.parameters(),
                        "lr": self.config.lr_sbert,
                    },
                    {
                        "params": self.model.projection.parameters(),
                        "lr": self.config.lr_b,
                    },
                    {
                        "params": self.model.clustering.parameters(),
                        "lr": self.config.lr_b,
                    },
                    {"params": self.model.cluster_pi, "lr": self.config.lr_gaussian},
                    {"params": self.model.cluster_mu, "lr": self.config.lr_gaussian},
                    {"params": self.model.cluster_sigma, "lr": self.config.lr_gaussian},
                    {
                        "params": self.model.encoder_mu.parameters(),
                        "lr": self.config.lr_t,
                    },
                    {
                        "params": self.model.encoder_sigma.parameters(),
                        "lr": self.config.lr_t,
                    },
                    {"params": self.model.decoder.parameters(), "lr": self.config.lr_t},
                ]
            )

    def train(self):
        self.model.train()

        if self.config.pre:
            self.train_pre()
        else:
            self.logger.log(self.config.str())

            self.logger.log("### train start")

            train_start = timer()

            last_prediction = None
            countdown = self.config.countdown
            num_iterations = len(self.train_loader)
            trigger = (
                np.linspace(1, 0, self.config.num_updates) ** 2
                * (self.config.num_epochs * num_iterations - 1)
            ).tolist()

            self.update(True)

            for epoch in range(self.config.num_epochs):
                epoch_start = timer()

                pre_command = (
                    "indexes = batch.pop()\n"
                    "batch.append(args[0].pseudo_labels[indexes])\n"
                    "batch.append(args[0].confident_masks[indexes])\n"
                )

                post_command = (
                    "step = args[1] + iteration\n"
                    "if step >= args[2][-1]:\n"
                    "    _ = args[2].pop()\n"
                    "    if step >= 100:\n"
                    "        args[3]()\n"
                    if self.config.logarithm
                    else ""
                )

                post_command = (
                    post_command + "temp = torch.min(args[4].cluster_pi.data)\n"
                    "if temp < 0:\n"
                    "    args[4].cluster_pi.data = args[4].cluster_pi.data - temp + 1\n"
                    "args[4].cluster_pi.data = args[4].cluster_pi.data / torch.sum(args[4].cluster_pi.data)\n"
                )

                # train
                output_sum, num_steps = train_loop(
                    self.train_loader,
                    lambda batch: [
                        self.squeeze_move(batch[0]),
                        self.squeeze_move(batch[1]),
                        self.squeeze_move(batch[2]),
                        batch[3].to(self.device).float(),
                        self.squeeze_move(batch[4]),
                        batch[5].to(self.device).float(),
                        batch[6].to(self.device),
                        batch[7].to(self.device),
                    ],
                    self.model.forward,
                    self.optimizer,
                    pre_command,
                    post_command,
                    self.data,  # args[0]
                    epoch * num_iterations,  # args[1]
                    trigger,  # args[2]
                    self.update,  # args[3]
                    self.model,  # args[4]
                )
                loss = output_sum["loss"] / num_steps

                # display
                if epoch % self.config.verbose_frequency == 0:
                    self.logger.log(
                        "epoch: %3d loss: %7.2f time: %s"
                        % (epoch, loss, timer(epoch_start))
                    )
                    self.logger.log(
                        "acc_b, nmi_b, acc_t, nmi_t: %7.2f %7.2f %7.2f %7.2f"
                        % (self.test())
                    )
                    self.logger.log(
                        "neighbor precision: %7.2f" % (self.test_precision())
                    )

                # save
                self.model.eval()

                prediction = test_loop(
                    self.test_loader,
                    lambda batch: [self.squeeze_move(batch[0])],
                    self.model.cluster_b,
                )

                self.model.train()

                temp = save(
                    last_prediction,
                    prediction,
                    self.config.epsilon,
                    self.logger,
                    countdown,
                    self.config.countdown,
                    self.model,
                    self.config.model_name,
                )

                if temp != None:
                    last_prediction, countdown = temp
                else:
                    break

                # update
                self.config.zeta = self.config.zeta_init * np.exp(
                    -self.config.zeta_decay * (epoch + 1)
                )

                if not self.config.logarithm:
                    self.update()

            self.logger.log("*** done")

            self.model.load_state_dict(
                torch.load("model/" + self.config.model_name + ".pt")
            )

            self.logger.log("final time: %s" % (timer(train_start)))
            self.logger.log(
                "acc_b, nmi_b, acc_t, nmi_t: %7.2f %7.2f %7.2f %7.2f" % (self.test())
            )
            self.logger.log("pre_b, pre_t: %7.2f %7.2f" % (self.test_precision(True)))

            self.logger.log("### train end")

            torch.save(
                self.model.state_dict(), "model/" + self.config.model_name + "-final.pt"
            )

        self.model.eval()

    def test(self):
        self.model.eval()

        y = np.array(self.data.labels)
        p_b = (
            test_loop(
                self.test_loader,
                lambda batch: [self.squeeze_move(batch[0])],
                self.model.cluster_b,
            )
            .cpu()
            .numpy()
        )
        p_t = (
            test_loop(
                self.test_loader,
                lambda batch: [batch[1].to(self.device).float()],
                self.model.cluster_t,
            )
            .cpu()
            .numpy()
        )

        acc_b = clustering_acc(p_b, y)
        nmi_b = clustering_nmi(p_b, y)
        acc_t = clustering_acc(p_t, y)
        nmi_t = clustering_nmi(p_t, y)

        self.model.train()

        return acc_b, nmi_b, acc_t, nmi_t

    def train_pre(self):
        self.model.train()

        self.logger.log(self.config.str())

        self.logger.log("### pre-train start")

        train_start = timer()

        last_prediction = None
        countdown = self.config.countdown

        for epoch in range(self.config.num_epochs):
            epoch_start = timer()

            # train
            output_sum, num_steps = train_loop(
                self.pre_loader,
                lambda batch: [
                    batch[0].to(self.device).float(),
                    batch[1].to(self.device).float(),
                ],
                self.model.forward_pre,
                self.optimizer,
            )
            loss = output_sum["loss"] / num_steps

            # display
            if epoch % self.config.verbose_frequency == 0:
                self.logger.log(
                    "epoch: %3d loss: %7.2f time: %s"
                    % (epoch, loss, timer(epoch_start))
                )
                self.logger.log(
                    "acc_kmeans, nmi_kmeans: %7.2f %7.2f" % (self.test_external())
                )
                self.logger.log("neighbor precision: %7.2f" % (self.test_precision()))

            # save
            temp = save(
                last_prediction,
                last_prediction,
                self.config.epsilon,
                self.logger,
                countdown,
                self.config.countdown,
                self.model,
                self.config.model_name,
            )

            if temp != None:
                last_prediction, countdown = temp
            else:
                break

        self.logger.log("*** done")

        self.model.load_state_dict(
            torch.load("model/" + self.config.model_name + ".pt")
        )

        self.logger.log("final time: %s" % (timer(train_start)))
        self.logger.log("acc_kmeans, nmi_kmeans: %7.2f %7.2f" % (self.test_external()))
        self.logger.log("neighbor precision: %7.2f" % (self.test_precision()))

        mu = self.test_external(True)
        self.model.define_mu(mu)
        self.model.to(self.device)

        self.logger.log("### pre-train end")

        torch.save(
            self.model.state_dict(), "model/" + self.config.model_name + "-pre.pt"
        )

        self.model.eval()

    def test_external(self, mean=False):
        self.model.eval()

        y = np.array(self.data.labels)
        z = test_loop(
            self.test_loader,
            lambda batch: [batch[1].to(self.device).float()],
            self.model.encode,
        )

        if mean:
            if (
                self.config.data_name == "agnews"
                or self.config.data_name == "biomedical"
                or self.config.data_name == "stackoverflow"
                or self.config.data_name == "webkb"
                or self.config.data_name == "reuters8"
                or self.config.data_name == "20newsgroups"
                or self.config.data_name == "bbc"
            ):
                mu, p = kmeans(
                    z, n_clusters=self.config.num_clusters, n_init=50, metric="ss"
                )
                mu = mu.cpu()
                p = p.cpu().numpy()
            elif self.config.data_name == "searchsnippets":
                mu, p = kmeans(
                    z, n_clusters=self.config.num_clusters, n_init=50, metric="dbs"
                )
                mu = mu.cpu()
                p = p.cpu().numpy()
            elif (
                self.config.data_name == "googlenews-s"
                or self.config.data_name == "googlenews-t"
                or self.config.data_name == "googlenews-ts"
                or self.config.data_name == "tweet"
            ):
                mu, p = ahc_sklearn(
                    z.cpu().numpy(), n_clusters=self.config.num_clusters
                )
                mu = torch.tensor(mu).float()

            acc = clustering_acc(p, y)
            nmi = clustering_nmi(p, y)

            self.logger.log("acc_init, nmi_init: %7.2f %7.2f" % (acc, nmi))
        else:
            _, p = kmeans(z, n_clusters=self.config.num_clusters)
            p = p.cpu().numpy()

            acc = clustering_acc(p, y)
            nmi = clustering_nmi(p, y)

            return acc, nmi

        return mu

    def test_precision(self, extra=False):
        self.model.eval()

        y = torch.tensor(self.data.labels).to(self.device)
        z = test_loop(
            self.test_loader,
            lambda batch: [batch[1].to(self.device).float()],
            self.model.encode,
        )
        n = knn(z, k=self.config.num_neighbors)

        precision = neighbor_precision(n, y)

        if extra:
            pre_t = precision

            y = torch.tensor(self.data.labels).to(self.device)
            f = test_loop(
                self.test_loader,
                lambda batch: [self.squeeze_move(batch[0])],
                self.model.project,
            )
            n = knn(f, k=self.config.num_neighbors)

            pre_b = neighbor_precision(n, y)

        self.model.train()

        if extra:
            return pre_b, pre_t
        else:
            return precision

    def update(self, boot=False):
        self.model.eval()

        z = test_loop(
            self.test_loader,
            lambda batch: [batch[1].to(self.device).float()],
            self.model.encode,
        )
        n_z = knn(z, k=self.config.num_neighbors).cpu().numpy().tolist()

        f = test_loop(
            self.test_loader,
            lambda batch: [self.squeeze_move(batch[0])],
            self.model.project,
        )
        n_f = knn(f, k=self.config.num_neighbors).cpu().numpy().tolist()

        self.train_dataset.texts_n = n_z
        self.train_dataset.tfidfs_n = n_f

        if boot:
            y = np.array(self.data.labels)
            p = test_loop(
                self.test_loader,
                lambda batch: [batch[1].to(self.device).float()],
                self.model.cluster_t,
            )

            acc = clustering_acc(p.cpu().numpy(), y)
            nmi = clustering_nmi(p.cpu().numpy(), y)

            self.logger.log("acc_init, nmi_init: %7.2f %7.2f" % (acc, nmi))

            self.data.pseudo_labels = p.cpu()

            if self.config.mask:
                s = test_loop(
                    self.test_loader,
                    lambda batch: [batch[1].to(self.device).float()],
                    self.model.score_t,
                )

                self.data.confident_masks = (
                    torch.max(s, dim=1)[0] > self.config.confidence
                ).cpu()
            else:
                self.data.confident_masks = torch.ones_like(
                    self.data.pseudo_labels
                ).bool()
        else:
            # special trick for pseudo-labeling
            self.model.train()

            s = test_loop(
                self.test_loader,
                lambda batch: [self.squeeze_move(batch[0])],
                self.model.score_b,
            )

            self.model.eval()

            if self.config.type == "ot":
                M = -torch.log(s)
                a = (torch.ones([M.size(0), 1]) / M.size(0)).to(self.device)
                b = (torch.ones([M.size(1), 1]) / M.size(1)).to(self.device)
                Pi = sinkhorn_knopp(M, a, b, epsilon=0.1)

                self.data.pseudo_labels = torch.argmax(Pi, dim=1).cpu()
            elif self.config.type == "aot":
                M = -torch.log(s)
                a = (torch.ones([M.size(0), 1]) / M.size(0)).to(self.device)
                b = (
                    (torch.ones([M.size(1), 1]) / M.size(1)).to(self.device)
                    if not hasattr(self, "b")
                    else self.b.to(self.device)
                )
                Pi, b = sinkhorn_knopp(
                    M,
                    a,
                    b,
                    epsilon=0.1,
                    variable=True,
                    regularization=self.config.regularization,
                    threshold=1e-5,
                )

                self.data.pseudo_labels = torch.argmax(Pi, dim=1).cpu()
                self.b = b.cpu()
            else:
                self.data.pseudo_labels = torch.argmax(s, dim=1).cpu()

            if self.config.mask:
                self.data.confident_masks = (
                    torch.max(s, dim=1)[0] > self.config.confidence
                ).cpu()

        self.model.train()

    def squeeze_move(self, texts):
        for key, value in texts.items():
            texts[key] = value.squeeze(1).to(self.device)

        return texts


###


def save(
    last_prediction,
    prediction,
    epsilon,
    logger,
    countdown,
    default_countdown,
    model,
    model_name,
):
    if (
        last_prediction == None
        or torch.sum(prediction != last_prediction) / last_prediction.size(0) > epsilon
    ):
        logger.log("*** prediction changed, copying")

        last_prediction = prediction
        countdown = default_countdown

        torch.save(model.state_dict(), "model/" + model_name + ".pt")
    else:
        logger.log("*** prediction nearly unchanged, skipping")

        countdown = countdown - 1

        if countdown == 0:
            return None

    return last_prediction, countdown


###
