"""
Based on https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
and https://github.com/ehoogeboom/multinomial_diffusion
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .utils import *

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianMultinomialDiffusion(torch.nn.Module):
    def __init__(
        self,
        num_classes: np.array,
        num_numerical_features: int,
        denoise_fn,
        num_timesteps=1000,
        gaussian_loss_type="mse",
        gaussian_parametrization="eps",
        multinomial_loss_type="vb_stochastic",
        parametrization="x0",
        scheduler="cosine",
        device=torch.device("cpu"),
    ):
        super(GaussianMultinomialDiffusion, self).__init__()
        assert multinomial_loss_type in ("vb_stochastic", "vb_all")
        assert parametrization in ("x0", "direct")

        if multinomial_loss_type == "vb_all":
            print(
                "Computing the loss using the bound on _all_ timesteps."
                " This is expensive both in terms of memory and computation."
            )

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes  # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate(
                [num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))]
            )
        ).to(device)

        self.slices_for_classes = [np.arange(self.num_classes[0])]
        offsets = np.cumsum(self.num_classes)
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(np.append([0], offsets)).to(device)

        self._denoise_fn = denoise_fn
        self.gaussian_loss_type = gaussian_loss_type
        self.gaussian_parametrization = gaussian_parametrization
        self.multinomial_loss_type = multinomial_loss_type
        self.num_timesteps = num_timesteps
        self.parametrization = parametrization
        self.scheduler = scheduler

        alphas = 1.0 - get_named_beta_schedule(scheduler, num_timesteps)
        alphas = torch.tensor(alphas.astype("float64"))
        betas = 1.0 - alphas

        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)

        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.tensor(np.append(1.0, alphas_cumprod[:-1]))
        alphas_cumprod_next = torch.tensor(np.append(alphas_cumprod[1:], 0.0))
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)

        # Gaussian diffusion

        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_log_variance_clipped = (
            torch.from_numpy(
                np.log(
                    np.append(self.posterior_variance[1], self.posterior_variance[1:])
                )
            )
            .float()
            .to(device)
        )
        self.posterior_mean_coef1 = (
            (betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
            .float()
            .to(device)
        )
        self.posterior_mean_coef2 = (
            (
                (1.0 - alphas_cumprod_prev)
                * np.sqrt(alphas.numpy())
                / (1.0 - alphas_cumprod)
            )
            .float()
            .to(device)
        )

        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.0e-5
        assert (
            log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item()
            < 1e-5
        )
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.0e-5

        # Convert to float32 and register buffers.
        self.register_buffer("alphas", alphas.float().to(device))
        self.register_buffer("log_alpha", log_alpha.float().to(device))
        self.register_buffer("log_1_min_alpha", log_1_min_alpha.float().to(device))
        self.register_buffer(
            "log_1_min_cumprod_alpha", log_1_min_cumprod_alpha.float().to(device)
        )
        self.register_buffer("log_cumprod_alpha", log_cumprod_alpha.float().to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.float().to(device))
        self.register_buffer(
            "alphas_cumprod_prev", alphas_cumprod_prev.float().to(device)
        )
        self.register_buffer(
            "alphas_cumprod_next", alphas_cumprod_next.float().to(device)
        )
        self.register_buffer(
            "sqrt_alphas_cumprod", sqrt_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            sqrt_one_minus_alphas_cumprod.float().to(device),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", sqrt_recip_alphas_cumprod.float().to(device)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            sqrt_recipm1_alphas_cumprod.float().to(device),
        )

        self.register_buffer("Lt_history", torch.zeros(num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(num_timesteps))

    # Gaussian part
    def gaussian_q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_1_min_cumprod_alpha, t, x_start.shape)
        return mean, variance, log_variance

    def gaussian_q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def gaussian_q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def gaussian_p_mean_variance(
        self,
        model_output,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_variance = torch.cat(
            [
                self.posterior_variance[1].unsqueeze(0).to(x.device),
                (1.0 - self.alphas)[1:],
            ],
            dim=0,
        )
        # model_variance = self.posterior_variance.to(x.device)
        model_log_variance = torch.log(model_variance)

        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        if self.gaussian_parametrization == "eps":
            pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        elif self.gaussian_parametrization == "x0":
            pred_xstart = model_output
        else:
            raise NotImplementedError

        model_mean, _, _ = self.gaussian_q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        ), f"{model_mean.shape}, {model_log_variance.shape}, {pred_xstart.shape}, {x.shape}"

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _vb_terms_bpd(
        self, model_output, x_start, x_t, t, clip_denoised=False, model_kwargs=None
    ):
        (
            true_mean,
            _,
            true_log_variance_clipped,
        ) = self.gaussian_q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.gaussian_p_mean_variance(
            model_output, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "out_mean": out["mean"],
            "true_mean": true_mean,
        }

    def _prior_gaussian(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.gaussian_q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def _gaussian_loss(self, model_out, x_start, x_t, t, noise, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}

        terms = {}
        if self.gaussian_loss_type == "mse":
            terms["loss"] = mean_flat((noise - model_out) ** 2)
        elif self.gaussian_loss_type == "kl":
            terms["loss"] = self._vb_terms_bpd(
                model_output=model_out,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]

        return terms["loss"]

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs={}):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def gaussian_p_sample(
        self,
        model_out,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        model_kwargs=None,
        cond_fn=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )

        # TODO: maybe to make stronger conditioning, I can make the variance 0 for those conditioned on known values.
        sample = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    # Multinomial part

    def multinomial_kl(self, log_prob1, log_prob2):
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):
        log_alpha_t = extract(self.log_alpha, t, log_x_t.shape)
        log_1_min_alpha_t = extract(self.log_1_min_alpha, t, log_x_t.shape)

        # alpha_t * E[xt] + (1 - alpha_t) 1 / K
        log_probs = log_add_exp(
            log_x_t + log_alpha_t,
            log_1_min_alpha_t - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def q_pred(self, log_x_start, t):
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_start.shape)
        log_1_min_cumprod_alpha = extract(
            self.log_1_min_cumprod_alpha, t, log_x_start.shape
        )

        log_probs = log_add_exp(
            log_x_start + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha - torch.log(self.num_classes_expanded),
        )

        return log_probs

    def predict_start(self, model_out, log_x_t, t, out_dict):
        # model_out = self._denoise_fn(x_t, t.to(x_t.device), **out_dict)

        assert model_out.size(0) == log_x_t.size(0)
        assert model_out.size(1) == self.num_classes.sum(), f"{model_out.size()}"

        log_pred = torch.empty_like(model_out)
        for ix in self.slices_for_classes:
            log_pred[:, ix] = F.log_softmax(model_out[:, ix], dim=1)
        return log_pred

    def q_posterior(self, log_x_start, log_x_t, t):
        # q(xt-1 | xt, x0) = q(xt | xt-1, x0) * q(xt-1 | x0) / q(xt | x0)
        # where q(xt | xt-1, x0) = q(xt | xt-1).

        # EV_log_qxt_x0 = self.q_pred(log_x_start, t)

        # print('sum exp', EV_log_qxt_x0.exp().sum(1).mean())
        # assert False

        # log_qxt_x0 = (log_x_t.exp() * EV_log_qxt_x0).sum(dim=1)
        t_minus_1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        t_minus_1 = torch.where(t_minus_1 < 0, torch.zeros_like(t_minus_1), t_minus_1)
        log_EV_qxtmin_x0 = self.q_pred(log_x_start, t_minus_1)

        num_axes = (1,) * (len(log_x_start.size()) - 1)
        t_broadcast = t.to(log_x_start.device).view(-1, *num_axes) * torch.ones_like(
            log_x_start
        )
        log_EV_qxtmin_x0 = torch.where(
            t_broadcast == 0, log_x_start, log_EV_qxtmin_x0.to(torch.float32)
        )

        # unnormed_logprobs = log_EV_qxtmin_x0 +
        #                     log q_pred_one_timestep(x_t, t)
        # Note: _NOT_ x_tmin1, which is how the formula is typically used!!!
        # Not very easy to see why this is true. But it is :)
        unnormed_logprobs = log_EV_qxtmin_x0 + self.q_pred_one_timestep(log_x_t, t)

        log_EV_xtmin_given_xt_given_xstart = unnormed_logprobs - sliced_logsumexp(
            unnormed_logprobs, self.offsets
        )

        return log_EV_xtmin_given_xt_given_xstart

    def p_pred(self, model_out, log_x, t, out_dict):
        if self.parametrization == "x0":
            log_x_recon = self.predict_start(model_out, log_x, t=t, out_dict=out_dict)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t
            )
        elif self.parametrization == "direct":
            log_model_pred = self.predict_start(
                model_out, log_x, t=t, out_dict=out_dict
            )
        else:
            raise ValueError
        return log_model_pred

    @torch.no_grad()
    def p_sample(self, model_out, log_x, t, out_dict):
        model_log_prob = self.p_pred(model_out, log_x=log_x, t=t, out_dict=out_dict)
        out = self.log_sample_categorical(model_log_prob)
        return out

    @torch.no_grad()
    def p_sample_loop(self, shape, out_dict):
        device = self.log_alpha.device

        b = shape[0]
        # start with random normal image.
        img = torch.randn(shape, device=device)

        for i in reversed(range(1, self.num_timesteps)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), out_dict
            )
        return img

    @torch.no_grad()
    def _sample(self, image_size, out_dict, batch_size=16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size), out_dict)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in reversed(range(0, t)):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long)
            )

        return img

    def log_sample_categorical(self, logits):
        full_sample = []
        for i in range(len(self.num_classes)):
            one_class_logits = logits[:, self.slices_for_classes[i]]
            uniform = torch.rand_like(one_class_logits)
            gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
            sample = (gumbel_noise + one_class_logits).argmax(dim=1)
            full_sample.append(sample.unsqueeze(1))
        full_sample = torch.cat(full_sample, dim=1)
        log_sample = index_to_log_onehot(full_sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def nll(self, log_x_start, out_dict):
        b = log_x_start.size(0)
        device = log_x_start.device
        loss = 0
        for t in range(0, self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()

            kl = self.compute_Lt(
                log_x_start=log_x_start,
                log_x_t=self.q_sample(log_x_start=log_x_start, t=t_array),
                t=t_array,
                out_dict=out_dict,
            )

            loss += kl

        loss += self.kl_prior(log_x_start)

        return loss

    def kl_prior(self, log_x_start):
        b = log_x_start.size(0)
        device = log_x_start.device
        ones = torch.ones(b, device=device).long()

        log_qxT_prob = self.q_pred(log_x_start, t=(self.num_timesteps - 1) * ones)
        log_half_prob = -torch.log(
            self.num_classes_expanded * torch.ones_like(log_qxT_prob)
        )

        kl_prior = self.multinomial_kl(log_qxT_prob, log_half_prob)
        return sum_except_batch(kl_prior)

    def compute_Lt(
        self, model_out, log_x_start, log_x_t, t, out_dict, detach_mean=False
    ):
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_x_t, t=t)
        log_model_prob = self.p_pred(model_out, log_x=log_x_t, t=t, out_dict=out_dict)

        if detach_mean:
            log_model_prob = log_model_prob.detach()

        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        kl = sum_except_batch(kl)

        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1.0 - mask) * kl

        return loss

    def sample_time(self, b, device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = (Lt_sqrt / Lt_sqrt.sum()).to(device)

            t = torch.multinomial(pt_all, num_samples=b, replacement=True).to(device)

            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def _multinomial_loss(self, model_out, log_x_start, log_x_t, t, pt, out_dict):
        if self.multinomial_loss_type == "vb_stochastic":
            kl = self.compute_Lt(model_out, log_x_start, log_x_t, t, out_dict)
            kl_prior = self.kl_prior(log_x_start)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior

            return vb_loss

        elif self.multinomial_loss_type == "vb_all":
            # Expensive, dont do it ;).
            # DEPRECATED
            return -self.nll(log_x_start)
        else:
            raise ValueError()

    def log_prob(self, x, out_dict):
        b, device = x.size(0), x.device
        if self.training:
            return self._multinomial_loss(x, out_dict)

        else:
            log_x_start = index_to_log_onehot(x, self.num_classes)

            t, pt = self.sample_time(b, device, "importance")

            kl = self.compute_Lt(
                log_x_start, self.q_sample(log_x_start=log_x_start, t=t), t, out_dict
            )

            kl_prior = self.kl_prior(log_x_start)

            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior

            return -loss

    def mixed_loss(self, x, out_dict, for_reconstruction=False, partial_table=None, known_features_mask=None):
        b = x.shape[0]
        device = x.device
        t, pt = self.sample_time(b, device, "uniform")

        x_num = x[:, : self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features :]
        original_data = torch.cat([x_num, x_cat], dim=1)

        x_num_t = x_num
        log_x_cat_t = x_cat
        if x_num.shape[1] > 0:
            noise = torch.randn_like(x_num)
            x_num_t = self.gaussian_q_sample(x_num, t, noise=noise)
        if x_cat.shape[1] > 0:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes)
            log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t)

        x_in = torch.cat([x_num_t, log_x_cat_t], dim=1)

        if for_reconstruction:
            x_in = original_data * known_features_mask[:b] + x_in * (1 - known_features_mask[:b])




        model_out = self._denoise_fn(x_in, t, **out_dict)

        model_out_num = model_out[:, : self.num_numerical_features]
        model_out_cat = model_out[:, self.num_numerical_features :]

        loss_multi = torch.zeros((1,)).float()
        loss_gauss = torch.zeros((1,)).float()
        # todo: Also modify input of _multinomial_loss() function for reconstruction attack training
        if x_cat.shape[1] > 0:
            loss_multi = self._multinomial_loss(
                model_out_cat, log_x_cat, log_x_cat_t, t, pt, out_dict
            ) / len(self.num_classes)

        if x_num.shape[1] > 0:
            # todo: make noise 0 for known_features?
            if for_reconstruction:
                loss_gauss = self._gaussian_loss(model_out_num, x_num, x_in[:, :self.num_numerical_features], t, noise * (1 - known_features_mask[:b]))
                # loss_gauss = self._gaussian_loss(model_out_num, x_num, x_in[:, :self.num_numerical_features], t, noise * (1 - known_features_mask[:b]) + model_out_num * known_features_mask[:b])
            else:
                loss_gauss = self._gaussian_loss(model_out_num, x_num, x_in[:, :self.num_numerical_features], t, noise)

        # loss_multi = torch.where(out_dict['y'] == 1, loss_multi, 2 * loss_multi)
        # loss_gauss = torch.where(out_dict['y'] == 1, loss_gauss, 2 * loss_gauss)

        return loss_multi.mean(), loss_gauss.mean()

    @torch.no_grad()
    def mixed_elbo(self, x0, out_dict):
        b = x0.size(0)
        device = x0.device

        x_num = x0[:, : self.num_numerical_features]
        x_cat = x0[:, self.num_numerical_features :]
        has_cat = x_cat.shape[1] > 0
        if has_cat:
            log_x_cat = index_to_log_onehot(x_cat.long(), self.num_classes).to(device)

        gaussian_loss = []
        xstart_mse = []
        mse = []
        mu_mse = []
        out_mean = []
        true_mean = []
        multinomial_loss = []
        for t in range(self.num_timesteps):
            t_array = (torch.ones(b, device=device) * t).long()
            noise = torch.randn_like(x_num)

            x_num_t = self.gaussian_q_sample(x_start=x_num, t=t_array, noise=noise)
            if has_cat:
                log_x_cat_t = self.q_sample(log_x_start=log_x_cat, t=t_array)
            else:
                log_x_cat_t = x_cat

            model_out = self._denoise_fn(
                torch.cat([x_num_t, log_x_cat_t], dim=1), t_array, **out_dict
            )

            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]

            kl = torch.tensor([0.0])
            if has_cat:
                kl = self.compute_Lt(
                    model_out=model_out_cat,
                    log_x_start=log_x_cat,
                    log_x_t=log_x_cat_t,
                    t=t_array,
                    out_dict=out_dict,
                )

            out = self._vb_terms_bpd(
                model_out_num,
                x_start=x_num,
                x_t=x_num_t,
                t=t_array,
                clip_denoised=False,
            )

            multinomial_loss.append(kl)
            gaussian_loss.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_num) ** 2))
            # mu_mse.append(mean_flat(out["mean_mse"]))
            out_mean.append(mean_flat(out["out_mean"]))
            true_mean.append(mean_flat(out["true_mean"]))

            eps = self._predict_eps_from_xstart(x_num_t, t_array, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        gaussian_loss = torch.stack(gaussian_loss, dim=1)
        multinomial_loss = torch.stack(multinomial_loss, dim=1)
        xstart_mse = torch.stack(xstart_mse, dim=1)
        mse = torch.stack(mse, dim=1)
        # mu_mse = torch.stack(mu_mse, dim=1)
        out_mean = torch.stack(out_mean, dim=1)
        true_mean = torch.stack(true_mean, dim=1)

        prior_gauss = self._prior_gaussian(x_num)

        prior_multin = torch.tensor([0.0])
        if has_cat:
            prior_multin = self.kl_prior(log_x_cat)

        total_gauss = gaussian_loss.sum(dim=1) + prior_gauss
        total_multin = multinomial_loss.sum(dim=1) + prior_multin
        return {
            "total_gaussian": total_gauss,
            "total_multinomial": total_multin,
            "losses_gaussian": gaussian_loss,
            "losses_multinimial": multinomial_loss,
            "xstart_mse": xstart_mse,
            "mse": mse,
            # "mu_mse": mu_mse
            "out_mean": out_mean,
            "true_mean": true_mean,
        }

    @torch.no_grad()
    def gaussian_ddim_step(
        self,
        model_out_num,
        x,
        t,
        clip_denoised=False,
        denoised_fn=None,
        eta=0.0,
        model_kwargs=None,
        cond_fn=None,
    ):
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=None,
        )

        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise

        return sample

    @torch.no_grad()
    def gaussian_ddim_sample(
        self, noise, T, out_dict, eta=0.0, model_kwargs=None, cond_fn=None
    ):
        x = noise
        b = x.shape[0]
        device = x.device
        for t in reversed(range(T)):
            print(f"Sample timestep {t:4d}", end="\r")
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_step(
                out_num, x, t_array, model_kwargs=model_kwargs, cond_fn=cond_fn
            )
        print()
        return x

    @torch.no_grad()
    def gaussian_ddim_reverse_step(
        self, model_out_num, x, t, clip_denoised=False, eta=0.0
    ):
        assert eta == 0.0, "Eta must be zero."
        out = self.gaussian_p_mean_variance(
            model_out_num,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=None,
            model_kwargs=None,
        )

        eps = (
            extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = extract(self.alphas_cumprod_next, t, x.shape)

        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_next)
            + torch.sqrt(1 - alpha_bar_next) * eps
        )

        return mean_pred

    @torch.no_grad()
    def gaussian_ddim_reverse_sample(
        self,
        x,
        T,
        out_dict,
    ):
        b = x.shape[0]
        device = x.device
        for t in range(T):
            print(f"Reverse timestep {t:4d}", end="\r")
            t_array = (torch.ones(b, device=device) * t).long()
            out_num = self._denoise_fn(x, t_array, **out_dict)
            x = self.gaussian_ddim_reverse_step(out_num, x, t_array, eta=0.0)
        print()

        return x

    @torch.no_grad()
    def multinomial_ddim_step(self, model_out_cat, log_x_t, t, out_dict, eta=0.0):
        # not ddim, essentially
        log_x0 = self.predict_start(
            model_out_cat, log_x_t=log_x_t, t=t, out_dict=out_dict
        )

        alpha_bar = extract(self.alphas_cumprod, t, log_x_t.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, t, log_x_t.shape)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        coef1 = sigma
        coef2 = alpha_bar_prev - sigma * alpha_bar
        coef3 = 1 - coef1 - coef2

        log_ps = torch.stack(
            [
                torch.log(coef1) + log_x_t,
                torch.log(coef2) + log_x0,
                torch.log(coef3) - torch.log(self.num_classes_expanded),
            ],
            dim=2,
        )

        log_prob = torch.logsumexp(log_ps, dim=2)

        out = self.log_sample_categorical(log_prob)

        return out

    @torch.no_grad()
    def sample_ddim(self, num_samples, y_dist, model_kwargs=None, cond_fn=None):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(self.num_classes_expanded)), device=device
            )
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {"y": y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict
            )
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_ddim_step(
                model_out_num,
                z_norm,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )
            if has_cat:
                log_z = self.multinomial_ddim_step(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def conditional_sample(self, ys, model_kwargs=None, cond_fn=None):
        device = self.log_alpha.device
        b = len(ys)
        z_norm = torch.randn((b, self.num_numerical_features), device=device)
        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()

        out_dict = {"y": ys.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict
            )
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_p_sample(
                model_out_num,
                z_norm,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def sample(self, num_samples, y_dist, model_kwargs=None, cond_fn=None):
        b = num_samples
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        if has_cat:
            uniform_logits = torch.zeros(
                (b, len(self.num_classes_expanded)), device=device
            )
            log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {"y": y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)
            model_out = self._denoise_fn(
                torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict
            )
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]
            z_norm = self.gaussian_p_sample(
                model_out_num,
                z_norm,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        print()
        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        sample = torch.cat([z_norm, z_cat], dim=1).cpu()
        return sample, out_dict

    @torch.no_grad()
    def reconstruct(self, b, y_dist,
        known_features_mask,
        known_features_values,
        model_kwargs=None, cond_fn=None
    ):
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        # if has_cat:
        #     uniform_logits = torch.zeros(
        #         (b, len(self.num_classes_expanded)), device=device
        #     )
        #     log_z = self.log_sample_categorical(uniform_logits)

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {"y": y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)

            # here's our novelty: replace z_norm features with known features.
            # Then, possibly, make t=0 for those features?
            z_norm_modified = z_norm * (1 - known_features_mask) + known_features_values * known_features_mask
            # t_modified = t * (1 - known_features_mask)

            model_out = self._denoise_fn(
                torch.cat([z_norm_modified, log_z], dim=1).float(), t, **out_dict
            )
            model_out_num = model_out[:, : self.num_numerical_features]
            model_out_cat = model_out[:, self.num_numerical_features :]

            # TODO: do I send in the partially-modified z_norm here too?
            z_norm = self.gaussian_p_sample(
                model_out_num,
                z_norm_modified,
                t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
            )["sample"]
            if has_cat:
                log_z = self.p_sample(model_out_cat, log_z, t, out_dict)

        z_ohe = torch.exp(log_z).round()
        z_cat = log_z
        if has_cat:
            z_cat = ohe_to_categories(z_ohe, self.num_classes)
        reconstruction = torch.cat([z_norm, z_cat], dim=1).cpu()
        return reconstruction, out_dict



    @torch.no_grad()
    def reconstruct_RePaint(self, b, y_dist,
        known_features_mask,
        known_features_values,
        resamples,
        jump,
        model_kwargs=None, cond_fn=None
    ):
        device = self.log_alpha.device
        z_norm = torch.randn((b, self.num_numerical_features), device=device)

        has_cat = self.num_classes[0] != 0
        log_z = torch.zeros((b, 0), device=device).float()
        # if has_cat:
        #     todo later

        y = torch.multinomial(y_dist, num_samples=b, replacement=True)
        out_dict = {"y": y.long().to(device)}
        for i in reversed(range(0, self.num_timesteps)):
            jump_ = jump(i)
            print(f"Sample timestep {i:4d}", end="\r")
            t = torch.full((b,), i, device=device, dtype=torch.long)

            x_num = known_features_values[:, : self.num_numerical_features]
            # x_cat = known_features_values[:, self.num_numerical_features:]

            z_norm_t = z_norm

            # (step 5, i.e. repeat steps 1 - 4 several times)
            for _ in range(resamples):
                jump_t = torch.full((b,), jump_, device=device, dtype=torch.long)
                # step 1
                noise = torch.randn_like(x_num)
                x_jump_t = self.gaussian_q_sample(x_num, jump_t, noise=noise)
                # if x_cat.shape[1] > 0:  # todo
                # x_t = torch.cat([x_num_t, log_x_cat_t], dim=1)

                # step 2
                model_out = self._denoise_fn(torch.cat([z_norm_t, log_z], dim=1).float(), jump_t, **out_dict)
                model_out_num = model_out[:, : self.num_numerical_features]
                # model_out_cat = model_out[:, self.num_numerical_features :]

                z_norm_jump_t = self.gaussian_p_sample(
                    model_out_num,
                    z_norm_t,
                    jump_t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                )["sample"]
                # if has_cat: # todo

                # step 3
                z_norm_jump_t = x_jump_t * known_features_mask + z_norm_jump_t * (1 - known_features_mask)

                # step 4
                noise = torch.randn_like(x_num)
                x_num_t = self.gaussian_q_sample(z_norm_jump_t, t, noise=noise)

                # if x_cat.shape[1] > 0: # todo
                # x_t = torch.cat([x_num_t, log_x_cat_t], dim=1)
                z_norm_t = x_num_t
                z_norm = z_norm_jump_t

        # z_ohe = torch.exp(log_z).round()
        # z_cat = log_z
        # if has_cat:
        #     z_cat = ohe_to_categories(z_ohe, self.num_classes)
        # reconstruction = torch.cat([z_norm, z_cat], dim=1).cpu()
        reconstruction = z_norm.cpu()
        return reconstruction, out_dict


    def sample_all(
        self,
        num_samples,
        batch_size,
        y_dist,
        ddim=False,
        model_kwargs=None,
        cond_fn=None,
    ):
        if ddim:
            print("Sample using DDIM.")
            sample_fn = self.sample_ddim
        else:
            sample_fn = self.sample

        b = batch_size

        all_y = []
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            sample, out_dict = sample_fn(
                b, y_dist, model_kwargs=model_kwargs, cond_fn=cond_fn
            )
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample[~mask_nan]
            out_dict["y"] = out_dict["y"][~mask_nan]

            all_samples.append(sample)
            all_y.append(out_dict["y"].cpu())
            if sample.shape[0] != b:
                raise FoundNANsError
            num_generated += sample.shape[0]

        x_gen = torch.cat(all_samples, dim=0)[:num_samples]
        y_gen = torch.cat(all_y, dim=0)[:num_samples]

        return x_gen, y_gen


    def reconstruct_all(
        self,
        batch_size,
        y_dist,
        known_features_mask,
        known_features_values,
        reconstruct_method_RePaint,
        resamples=None,
        jump=None,
        ddim=False,
        model_kwargs=None,
        cond_fn=None,
    ):
        b = batch_size
        N = len(known_features_values)

        device = self.log_alpha.device
        known_features_mask = known_features_mask.to(device)
        known_features_values = known_features_values.to(device)

        all_y = []
        full_reconstruction = []
        i = 0
        while i < N:
            b_ = min(b, N-i)
            if reconstruct_method_RePaint:
                reconstruction, out_dict = self.reconstruct_RePaint(
                    b_, y_dist,
                    known_features_mask[i:i+b_],
                    known_features_values[i:i+b_],
                    resamples,
                    jump,
                    model_kwargs=model_kwargs, cond_fn=cond_fn
                )
            else:
                reconstruction, out_dict = self.reconstruct(
                    b_, y_dist,
                    known_features_mask[i:i+b_],
                    known_features_values[i:i+b_],
                    model_kwargs=model_kwargs, cond_fn=cond_fn
                )
            mask_nan = torch.any(reconstruction.isnan(), dim=1)
            # reconstruction = reconstruction[~mask_nan]
            # out_dict["y"] = out_dict["y"][~mask_nan]

            full_reconstruction.append(reconstruction)
            all_y.append(out_dict["y"].cpu())
            # if mask_nan.sum().item() > 0:
            #     raise FoundNANsError
            i += reconstruction.shape[0]

        x_gen = torch.cat(full_reconstruction, dim=0)
        y_gen = torch.cat(all_y, dim=0)

        return x_gen, y_gen

    #
    # @torch.no_grad()
    # def conditional_sample_with_known_features(self, y_dist, known_features_mask, known_features_values,
    #                                            scale=10.0):
    #     """
    #     Generate samples with specific feature values fixed.
    #
    #     Args:
    #         num_samples: Number of samples to generate
    #         y_dist: Distribution of class labels (if applicable)
    #         known_features_mask: Binary mask (1=known, 0=unknown) [num_samples, num_features]
    #         known_features_values: Values of known features [num_samples, num_features]
    #         scale: Conditioning strength
    #     """
    #     b = len(known_features_mask)
    #     device = self.log_alpha.device
    #
    #     # Initialize with random noise
    #     z_norm = torch.randn((b, self.num_numerical_features), device=device)
    #
    #     # Set up categorical features if present
    #     has_cat = self.num_classes[0] != 0
    #     log_z = torch.zeros((b, 0), device=device).float()
    #     # if has_cat:
    #     #     uniform_logits = torch.zeros((b, len(self.num_classes_expanded)), device=device)
    #     #     log_z = self.log_sample_categorical(uniform_logits)
    #
    #     # Create label conditioning
    #     y = torch.multinomial(y_dist, num_samples=b, replacement=True)
    #     out_dict = {"y": y.long().to(device)}
    #
    #     # Split masks for numerical and categorical features
    #     num_mask = known_features_mask[:, :self.num_numerical_features]
    #     num_values = known_features_values[:, :self.num_numerical_features]
    #
    #     # # Handle categorical masks (more complex due to one-hot encoding)
    #     # cat_mask = None
    #     # cat_values = None
    #     # if has_cat:
    #     #     cat_mask = known_features_mask[:, self.num_numerical_features:]
    #     #     cat_values = known_features_values[:, self.num_numerical_features:]
    #
    #     # Define conditioning function - we'll only apply it to numerical features for simplicity
    #     def cond_fn(x, t_):
    #         return self.partial_features_cond_fn(x, t_, num_mask, num_values, scale)
    #
    #     # Sampling loop
    #     for i in reversed(range(0, self.num_timesteps)):
    #         print(f"Sample timestep {i:4d}", end="\r")
    #         t = torch.full((b,), i, device=device, dtype=torch.long)
    #
    #         # Get model prediction
    #         model_out = self._denoise_fn(torch.cat([z_norm, log_z], dim=1).float(), t, **out_dict)
    #         model_out_num = model_out[:, :self.num_numerical_features]
    #         model_out_cat = model_out[:, self.num_numerical_features:]
    #
    #         # TODO: try making the known values of z_norm the true values. Could that make even stronger conditioning?
    #         # Apply denoising step with conditioning
    #         z_norm = self.gaussian_p_sample(
    #             model_out_num,
    #             z_norm,
    #             t,
    #             clip_denoised=False,
    #             cond_fn=cond_fn,
    #             model_kwargs={},
    #         )["sample"]
    #
    #         # # Process categorical features if present
    #         # if has_cat:
    #         #     log_z = self.p_sample(model_out_cat, log_z, t, out_dict)
    #
    #         # For numerical features that are known, directly enforce the noisy values
    #         # This ensures stronger conditioning than just the gradient
    #         if i > 0:  # Skip last step to get clean output
    #             # Add appropriately scaled noise to known features at this timestep
    #             noise_level = extract(self.sqrt_one_minus_alphas_cumprod, t, z_norm.shape)
    #             alpha_level = extract(self.sqrt_alphas_cumprod, t, z_norm.shape)
    #
    #             # Create noisy version of ground truth
    #             noisy_target = num_values * alpha_level + torch.randn_like(num_values) * noise_level
    #
    #             # Apply mask to replace only known features
    #             z_norm = torch.where(num_mask.bool(), noisy_target, z_norm)
    #
    #     print()
    #     # # Process final output
    #     # z_ohe = torch.exp(log_z).round() if has_cat else log_z
    #     # z_cat = ohe_to_categories(z_ohe, self.num_classes) if has_cat else log_z
    #     sample = torch.cat([z_norm], dim=1).cpu()
    #
    #     return sample, out_dict
    #
    # def partial_features_cond_fn(self, x, t, known_features_mask, known_features_values, scale=1.0):
    #     """
    #     Conditioning function that guides the diffusion model to respect known feature values.
    #
    #     Args:
    #         x: Current tensor being denoised
    #         t: Current timestep
    #         known_features_mask: Binary mask indicating which features are known (1) vs unknown (0)
    #         known_features_values: Values of the known features (same shape as x)
    #         scale: Strength of the conditioning (higher = stronger conditioning)
    #
    #     Returns:
    #         Gradient to guide the denoising process
    #     """
    #     # Calculate noisy version of known features at current timestep
    #     with torch.enable_grad():
    #         x_in = x.detach().requires_grad_(True)
    #         # Compute the mean squared error between the denoised known features and true known features
    #         noise_level = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
    #         target_noisy = known_features_values * extract(self.sqrt_alphas_cumprod, t, x.shape) + torch.randn_like(known_features_values) * noise_level
    #         error = ((x_in - target_noisy) * known_features_mask) ** 2
    #         grad = torch.autograd.grad(error.sum(), x_in)[0]
    #         return -scale * grad
