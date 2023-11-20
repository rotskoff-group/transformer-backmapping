import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical, Independent, MixtureSameFamily


class GenModel:
    """A generative model base class that can be used 
       for sampling and computing log probabilities.
    """
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError

    def log_prob(self):
        raise NotImplementedError

    def save(self, epoch_num=None):
        pass

    def load(self, epoch_num=None):
        pass


class IdentityModel(GenModel):
    """A GenModel that returns the input as the output and 0 as the log probability.
    """

    def __init__(self, all_samples):
        """
        Arguments:
            all_samples {torch.Tensor} -- All samples
        """
        self.all_samples = all_samples

    def sample(self, n_samples, c_info=None):
        """
        Arguments:
            n_samples {int} -- Number of samples to return and is equal to all_samples.shape[0]
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            tuple -- (all_samples, log_prob) where all_samples is a torch.Tensor of shape (n_samples, all_samples.shape[1])
                     and log_prob is a torch.Tensor of shape (n_samples,) with all 0s.
        """
        assert n_samples == self.all_samples.shape[0]
        return self.all_samples, torch.zeros(n_samples,
                                             device=self.all_samples.device)

    def log_prob(self, z):
        """Computes the "log probability" of the input z
        Arguments:
            z {torch.Tensor} -- Input
        Returns:
            torch.Tensor -- A torch.Tensor of shape (z.shape[0],) with all 0s.
        """
        return torch.zeros(z.shape[0], device=z.device)


class DeltaModel(GenModel):
    """A GenModel that returns a delta distribution centered at all_centers.
    """
    def __init__(self, all_centers):
        """
        Arguments:
            all_centers {torch.Tensor} -- All centers
        """
        self.all_centers = all_centers

    def sample(self, n_samples, c_info=None):
        """
        Arguments:
            n_samples {int} -- Number of samples to return and is equal to all_centers.shape[0]
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            tuple -- (all_centers, log_prob) where all_centers is a torch.Tensor of shape (n_samples, all_centers.shape[1])
                     and log_prob is a torch.Tensor of shape (n_samples,) with all 0s.
        """
        return self.all_centers.unsqueeze(0).repeat(n_samples, 1), torch.zeros(n_samples,
                                                                               device=self.all_centers.device)

    def log_prob(self, z):
        """Computes the "log probability" of the input z
        Arguments:
            z {torch.Tensor} -- Input
        Returns:
            torch.Tensor -- A torch.Tensor of shape (z.shape[0],) with all 0s.
        """
        return torch.zeros(z.shape[0], device=z.device)


class GaussianModel(GenModel):
    """A GenModel that returns a Gaussian distribution
        centered at all_means with all_vars as the variance.
        Device of all_means and all_vars is assumed to be the same 
        and the desired one.
    """
    def __init__(self, all_means, all_vars):
        """
        Arguments:
            all_means {torch.Tensor} -- All means
            all_vars {torch.Tensor} -- All variances
        """
        self.all_means = all_means
        self.all_vars = all_vars
        self.gaussian = torch.distributions.MultivariateNormal(loc=all_means,
                                                               covariance_matrix=torch.diag(all_vars))

    def sample(self, n_samples, c_info=None):
        """
        Arguments:
            n_samples {int} -- Number of samples to return
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            tuple -- (sample, log_prob) where sample is a torch.Tensor of shape (n_samples, all_means.shape[1])
                     and log_prob is a torch.Tensor of shape (n_samples,) with the log probability of the sample.
        """
        sample = self.gaussian.sample((n_samples,))
        log_prob = self.gaussian.log_prob(sample)
        return sample, log_prob

    def log_prob(self, z, c_info=None):
        """Arguments:
            z {torch.Tensor} -- Input
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            torch.Tensor -- A torch.Tensor of shape (z.shape[0],) with the log probability of the input z.
        """
        return self.gaussian.log_prob(z)



class ConcatenatedModel(GenModel):
    """Given a list of GenModels, samples from each model and concatenates the samples.
    """

    def __init__(self, all_models, split_sections=None):
        """
        Arguments:
            all_models {list} -- A list of GenModels
            split_sections {list} -- A list of integers that specify how to split the samples from all_models
            into distances and angles.
        """
        self.all_models = all_models
        self.split_sections = split_sections

    def sample(self, n_samples, c_info=[]):
        """Samples from each model and concatenates the samples.
        Arguments:
            n_samples {int} -- Number of samples to return
            c_info {list, None} -- A list of conditional information for each model
        Returns:
            tuple -- (all_samples, log_prob) where all_samples is a torch.Tensor of shape (n_samples, sum(split_sections))
                        and log_prob is a torch.Tensor of shape (n_samples,) with the log probability of the sample.
        """
        if isinstance(c_info, list):
            assert len(c_info) == len(self.all_models)
        else:
            c_info = [c_info] * len(self.all_models)
        all_samples = []
        all_log_probs = []
        for (c_info_i, model) in zip(c_info, self.all_models):
            samples, log_probs = model.sample(n_samples, c_info=c_info_i)
            all_samples.append(samples)
            all_log_probs.append(log_probs)
        all_samples = torch.cat(all_samples, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=-1).sum(-1)
        return all_samples, all_log_probs

    def log_prob(self, all_x, c_info=[]):
        """Splits all_x into samples from each constituent GenModel and computes the log probability of each part.
        Arguments:
            all_x {torch.Tensor} -- Input
            c_info {list, None} -- A list of conditional information for each model
        """
        if isinstance(c_info, list):
            assert len(c_info) == len(self.all_models)
        else:
            c_info = [c_info] * len(self.all_models)

        all_x = torch.tensor_split(all_x, self.split_sections,
                                   dim=-1)
        all_log_prob = 0
        for (x, model, c_info_i) in zip(all_x, self.all_models, c_info):
            log_prob = model.log_prob(x, c_info=c_info_i)
            all_log_prob += log_prob
        return all_log_prob


class GaussianMixtureModel(nn.Module):
    """A Gaussian mixture model that can be used for sampling and computing log probabilities.
        Model is not meant to be trained
    """

    def __init__(self, means, covs, weights, all_offset,
                 folder_name="./", sample_more_info=False, device=torch.device("cuda:0")):
        """
        Arguments:
            means {torch.Tensor} -- Means of the Gaussian mixture model
            covs {torch.Tensor} -- Covariances of the Gaussian mixture model
            weights {torch.Tensor} -- Weights of the Gaussian mixture model
            all_offset {torch.Tensor} -- Offset for the samples (used when a mode is split at the boundary (e.g. -pi and pi))
            folder_name {str} -- Folder name to save the model
            sample_more_info {bool} -- Whether to return more information when sampling
            device {torch.device} -- Device to use
        """
        super().__init__()
        self.register_buffer('means', means)
        self.register_buffer('covs', covs)
        self.register_buffer('weights', weights)
        self.register_buffer('all_offset',
                             torch.tensor(all_offset, device=device))
        self.folder_name = folder_name
        self.sample_more_info = sample_more_info
        self.device = device
        assert self.weights.shape[0] == self.means.shape[0] == self.covs.shape[0]

        component_distributions = Independent(MultivariateNormal(self.means,
                                                                 self.covs), 0)
        mixture_distribution = Categorical(self.weights)
        self.gmm = MixtureSameFamily(mixture_distribution,
                                     component_distributions)
        
    def log_prob(self, all_x, c_info=None):
        """Given a batch of samples, computes the log probability of each sample.
        Arguments:
            all_x {torch.Tensor} -- Input
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            torch.Tensor -- A torch.Tensor of shape (all_x.shape[0],) with the log probability of each sample.
        """
        offset_indices = torch.where(self.all_offset[0] != 0)[0]
        all_x[:, offset_indices] += self.all_offset[:, offset_indices]
        to_subtract = (all_x[:, offset_indices] > torch.pi) * 2 * torch.pi
        to_add = (all_x[:, offset_indices] < -torch.pi) * 2 * torch.pi
        all_x[:, offset_indices] -= to_subtract
        all_x[:, offset_indices] += to_add

        log_prob = self.gmm.log_prob(all_x)

        return log_prob

    def _sample_more_info(self, sample_shape):
        """Samples from the Gaussian mixture model and returns the sample and the log probability of the mixture component.
        Arguments:
            sample_shape {tuple} -- Shape of the sample
        Returns:
            tuple -- (sample, mix_dist_info) where sample is a torch.Tensor of shape (sample_shape) 
            and mix_dist_info is a torch.Tensor of shape (sample_shape, 2) with the sample and the log probability of the mixture component.
        """
        mix_sample = self.gmm.mixture_distribution.sample(sample_shape)
        mix_log_prob = self.gmm.mixture_distribution.log_prob(mix_sample)
        mix_dist_info = torch.stack((mix_sample, mix_log_prob), dim=-1)
        comp_sample = self.gmm.component_distribution.sample(sample_shape)
        sample_len = len(sample_shape)
        batch_len = len(self.gmm.batch_shape)
        gather_dim = sample_len + batch_len

        mix_shape = mix_sample.shape
        es = self.gmm.event_shape
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(torch.Size(
            [1] * len(mix_shape)) + torch.Size([1]) + es)
        samples = torch.gather(comp_sample, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim), mix_dist_info
    
    def sample_from_selected_components(self, mix_sample, c_info=None):
        """Given a batch of mixture components, samples from the selected components.
        Arguments:
            mix_sample {torch.Tensor} -- Input
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            torch.Tensor -- A torch.Tensor of shape (mix_sample.shape[0], self.means.shape[1]) with the samples.
        """
        n_samples = mix_sample.shape[0]
        sample_shape = (n_samples,)
        comp_sample = self.gmm.component_distribution.sample(sample_shape)
        sample_len = len(sample_shape)
        batch_len = len(self.gmm.batch_shape)
        gather_dim = sample_len + batch_len


        mix_shape = mix_sample.shape
        es = self.gmm.event_shape
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(torch.Size(
            [1] * len(mix_shape)) + torch.Size([1]) + es)
        samples = torch.gather(comp_sample, gather_dim, mix_sample_r)
        samples = samples.squeeze(gather_dim)


        offset_indices = torch.where(self.all_offset[0] != 0)[0]
        samples[:, offset_indices] -= self.all_offset[:, offset_indices]
        to_subtract = (samples[:, offset_indices] > torch.pi) * 2 * torch.pi
        to_add = (samples[:, offset_indices] < -torch.pi) * 2 * torch.pi
        samples[:, offset_indices] -= to_subtract
        samples[:, offset_indices] += to_add
        return samples




    def sample(self, n_samples, c_info=None):
        """Samples from the Gaussian mixture model.
        Arguments:
            n_samples {int} -- Number of samples to return
            c_info {torch.Tensor} -- Conditional information (not used)
        Returns:
            tuple -- (sample, log_prob) where sample is a torch.Tensor of shape (n_samples, self.means.shape[1]) 
            and log_prob is a torch.Tensor of shape (n_samples,) with the log probability of the sample.
        """
        if self.sample_more_info:
            sample, mix_dist_info = self._sample_more_info((n_samples,))
        else:
            sample = self.gmm.sample((n_samples,))
        log_prob = self.gmm.log_prob(sample)
        offset_indices = torch.where(self.all_offset[0] != 0)[0]
        sample[:, offset_indices] -= self.all_offset[:, offset_indices]
        to_subtract = (sample[:, offset_indices] > torch.pi) * 2 * torch.pi
        to_add = (sample[:, offset_indices] < -torch.pi) * 2 * torch.pi
        sample[:, offset_indices] -= to_subtract
        sample[:, offset_indices] += to_add
        if self.sample_more_info:
            sample = torch.cat((sample, mix_dist_info), axis=-1)
        return sample, log_prob

    def save(self, epoch_num=None):
        """Saves the model
        Arguments:
            epoch_num {int, None} -- Epoch number (used in the file name)
        Saves:
            model_state_dict {dict} -- Dictionary containing the model state dict and the epoch number
        """
        model_to_save = {"model_state_dict": self.state_dict(),
                         "epoch_num": epoch_num}
        torch.save(model_to_save,
                   self.folder_name + "flow")
        if epoch_num is not None:
            torch.save(model_to_save,
                       self.folder_name + "flow_" + str(epoch_num))

    def load(self, epoch_num=None):
        """Loads the model
        Arguments:
            epoch_num {int, None} -- Epoch number (used in the file name)
        Returns:
            epoch_num {int} -- Epoch number
        """
        if epoch_num is None:
            flow_checkpoint = torch.load(self.folder_name + "flow",
                                         map_location=self.device)
        else:
            flow_checkpoint = torch.load(self.folder_name + "flow_" + str(epoch_num),
                                         map_location=self.device)
        self.load_state_dict(flow_checkpoint["model_state_dict"])
        component_distributions = Independent(MultivariateNormal(self.means,
                                                                 self.covs), 0)
        mixture_distribution = Categorical(self.weights)
        self.gmm = MixtureSameFamily(mixture_distribution,
                                     component_distributions)
        return flow_checkpoint["epoch_num"]


