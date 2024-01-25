import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_models
import random

# Partially based on https://github.com/ermongroup/CSDI
#  (MIT license)


class DiffRI_base(nn.Module):
    def __init__(self, target_dim, config, device, density):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.config = config
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.density = density
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_models(config, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(
            self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_inter_mask(self, observed_mask, init_step=5):
        B, K, L = observed_mask.shape
        cond_mask = torch.zeros_like(observed_mask)
        target_mask = torch.zeros_like(observed_mask)
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        target_list = torch.zeros(B)
        for i in range(B):
            target_k = random.sample(range(K), 1)
            target_list[i] = target_k[0]
            for j in range(K):
                if j == target_k[0]:
                    sample_ratio = 0.5
                else:
                    sample_ratio = 0
                num_observed = sum(observed_mask[i, j, :])
                num_masked = (num_observed * sample_ratio).round()
                rand_for_mask[i, j, :][rand_for_mask[i, j].topk(
                    int(num_masked)).indices] = -1
                cond_mask[i, j, :][rand_for_mask[i, j] > 0] = 1

            target_mask[i, target_k[0],
                        :][rand_for_mask[i, target_k[0], :] == -1] = 1
        return cond_mask == 1, target_mask == 1, target_list

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(
            observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(
            0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, epoch_no, target_list, target_mask=None, set_t=-1
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t, target_mask=target_mask, target_list=target_list, epoch_no=epoch_no)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, epoch_no, target_list, target_mask=None, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)  np.cumprod(1-self.beta)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + \
            (1.0 - current_alpha) ** 0.5 * noise
        if target_mask == None:
            target_mask = observed_mask - cond_mask
        total_input = self.set_input_to_diffmodel(
            noisy_data, observed_data, cond_mask, target_mask)
        predicted, l1_loss = self.diffmodel(
            total_input, side_info, t, target_list, epoch_no=epoch_no)  # (B,K,L)
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        l1_loss = torch.sum(l1_loss) / torch.numel(l1_loss)

        if self.config["exp_set"]["no-reg"]:
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        else:
            loss = (residual ** 2).sum() / (num_eval if num_eval >
                                            0 else 1) + 0.01 * torch.abs(l1_loss - self.density)
        return loss

    def compute_alpha(self, beta, t):
        beta = torch.tensor(beta).to(self.device)
        beta = torch.cat([torch.zeros(1).to(self.device), beta], dim=0)

        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def generalized_steps(self, x, cond_mask, target_mask, side_info, seq, b, **kwargs):
        with torch.no_grad():
            n = x.size(0)
            seq_next = [-1] + list(seq[:-1])
            xt = torch.randn_like(x)
            xs = [xt.unsqueeze(1)]
            cond_data = (cond_mask * x).unsqueeze(1)
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = self.compute_alpha(b, t.long())
                at_next = self.compute_alpha(b, next_t.long())
                xt = xs[-1]  # .to('cuda')
                noisy_target = (target_mask.unsqueeze(1) * xt)
                diff_input = torch.cat([cond_data, noisy_target], dim=1)

                et, _ = self.diffmodel(diff_input.to(dtype=torch.float), side_info, torch.tensor(
                    [i]).to(self.device), target_list=kwargs['target_list'], epoch_no=None)
                x0_t = (xt - et.unsqueeze(1) * (1 - at).sqrt()) / at.sqrt()
                c1 = (
                    kwargs.get("eta", 0) * ((1 - at / at_next)
                                            * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()

                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x.unsqueeze(1)) + \
                    c2 * et.unsqueeze(1)
                xs.append(xt_next.to('cpu'))

        return xs[-1]

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask, target_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = (target_mask * noisy_data).unsqueeze(1)
            total_input = torch.cat(
                [cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def impute(self, observed_data, cond_mask, target_mask, side_info, n_samples, target_list):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * \
                        noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            if self.config['model']['sampling'] == 'DDPM':
                for t in range(self.num_steps - 1, -1, -1):
                    if self.is_unconditional == True:
                        diff_input = cond_mask * \
                            noisy_cond_history[t] + \
                            (1.0 - cond_mask) * current_sample
                        diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                    else:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = (
                            target_mask * current_sample).unsqueeze(1)
                        diff_input = torch.cat(
                            [cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    et, _ = self.diffmodel(diff_input.to(dtype=torch.float), side_info, torch.tensor(
                        [t]).to(self.device), target_list=target_list, epoch_no=None)

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / \
                        (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * et)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = (
                            (1.0 - self.alpha[t - 1]) /
                            (1.0 - self.alpha[t]) * self.beta[t]
                        ) ** 0.5
                        current_sample += sigma * noise

                imputed_samples[:, i] = current_sample.detach()

            elif self.config['model']['sampling'] == 'DDIM':

                if self.config['model']['schedule'] == "uniform":
                    skip = self.num_steps // self.config['model']['sample_step']
                    seq = range(0, self.num_steps, skip)
                elif self.config['model']['schedule'] == "quad":
                    seq = (
                        np.linspace(
                            0, np.sqrt(self.num_steps *
                                       0.8), self.config['model']['sample_step']
                        )
                        ** 2
                    )
                    seq = [int(s) for s in list(seq)]
                else:
                    raise NotImplementedError

                xs = self.generalized_steps(
                    observed_data, cond_mask, target_mask, side_info, seq, self.beta, eta=0.0, target_list=target_list)
                xs = xs.squeeze(1)
                imputed_samples[:, i] = xs.detach()

        return imputed_samples

    def forward(self, batch, epoch_no, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)

        if self.target_strategy == "customed":
            cond_mask, target_mask, target_list = self.get_inter_mask(
                observed_mask)
        else:
            raise NotImplementedError("Masking stretegy not implemented")
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train, epoch_no=epoch_no, target_mask=target_mask, target_list=target_list)

    def evaluate(self, batch, s_list, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)
        with torch.no_grad():
            not_s_list = [i for i in range(
                observed_data.shape[1]) if i not in s_list]
            gt_mask[:, not_s_list, :] = 1
            gt_mask *= observed_mask
            cond_mask = gt_mask
            cond_mask[:, not_s_list] = observed_mask[:, not_s_list]
            target_mask = observed_mask - cond_mask
            target_mask[:, not_s_list, :] = 0
            side_info = self.get_side_info(observed_tp, cond_mask)

            B = cond_mask.shape[0]
            target_list = torch.ones(B) * s_list.item()
            samples = self.impute(observed_data, cond_mask=cond_mask, target_mask=target_mask,
                                  side_info=side_info, n_samples=n_samples, target_list=target_list)

        return samples, observed_data, target_mask, observed_mask, observed_tp


class DiffRI(DiffRI_base):
    def __init__(self, config, device, target_dim=10, density=0.5):
        super().__init__(target_dim=target_dim, config=config, device=device, density=density)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float().clone()
        observed_mask = batch["observed_mask"].to(self.device).float().clone()
        observed_tp = batch["timepoints"].to(self.device).float().clone()
        gt_mask = batch["gt_mask"].to(self.device).float().clone()
        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
