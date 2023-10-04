import numpy as np
import torch

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):
    """Estimate the full covariance matrices.
    Parameters
    ----------
    resp : array-like of shape (n_samples, n_components)
    X : array-like of shape (n_samples, n_features)
    nk : array-like of shape (n_components,)
    means : array-like of shape (n_components, n_features)
    reg_covar : float
    Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
        The covariance matrix of the current components.
    """
    '''
    rr = np.array(resp)
    xx = np.array(X)
    nn = np.array(nk)
    mm = np.array(means)
    
    n_components, n_features = mm.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = xx - mm[k]
        covariances[k] = np.dot(rr[:, k] * diff.T, diff) / nn[k]
        covariances[k].flat[:: n_features + 1] += reg_covar
        print("ooooooooo   ", covariances[k].flat[:: n_features + 1].shape)
        
    return covariances
    '''
    
    n_components, n_features = means.shape
    covariances = torch.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = torch.matmul(resp[:, k] * diff.t(), diff) / nk[k]
        covariances[k].view(covariances[k].numel())[:: n_features + 1] += reg_covar
    return covariances
    
    
def _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar):
    """Estimate the diagonal covariance vectors.
    Parameters
    ----------
    responsibilities : array-like of shape (n_samples, n_components)
    X : array-like of shape (n_samples, n_features)
    nk : array-like of shape (n_components,)
    means : array-like of shape (n_components, n_features)
    reg_covar : float
    Returns
    -------
    covariances : array, shape (n_components, n_features)
        The covariance vector of the current components.
    """
    
    avg_X2 = torch.matmul(resp.t(), X * X) / nk.clone().unsqueeze(-1)  #torch.ones(7,1).cuda()  #nk.unsqueeze(-1)  #nk[:, np.newaxis]
    
    avg_means2 = means ** 2
    avg_X_means = means * torch.matmul(resp.t(), X) / nk.clone().unsqueeze(-1)  #torch.ones(7,1).cuda()  #nk.unsqueeze(-1)  #nk[:, np.newaxis]
    
    return avg_X2 - 2 * avg_X_means + avg_means2 + reg_covar
    
def _estimate_gaussian_parameters(X, resp, reg_covar):
    """Estimate the Gaussian distribution parameters.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data array.
    resp : array-like of shape (n_samples, n_components)
        The responsibilities for each data sample in X.
    reg_covar : float
        The regularization added to the diagonal of the covariance matrices.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    nk : array-like of shape (n_components,)
        The numbers of data samples in the current components.
    means : array-like of shape (n_components, n_features)
        The centers of the current components.
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    """
    nk = resp.sum(dim=0) + 10 * np.finfo(np.float32).eps
    
    means = torch.matmul(resp.t(), X) / nk.clone().unsqueeze(-1)  #torch.ones(7,1).cuda()  #nk.unsqueeze(-1)   #nk[:, torch.newaxis]
    
    covariances = _estimate_gaussian_covariances_diag(resp, X, nk, means, reg_covar)
    #covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    '''covariances = {"full": _estimate_gaussian_covariances_full,
                   "tied": _estimate_gaussian_covariances_tied,
                   "diag": _estimate_gaussian_covariances_diag,
                   "spherical": _estimate_gaussian_covariances_spherical
                   }[covariance_type](resp, X, nk, means, reg_covar)'''
    return nk, means, covariances

def m_step(X, log_resp):
        """M step.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        reg_covar = 1e-6
        n_samples, _ = X.shape
        weights_, means_, covariances_ = (
            _estimate_gaussian_parameters(X, torch.exp(log_resp), reg_covar))
        weights_ /= n_samples
        
        #precisions_cholesky_ = _compute_precision_cholesky(
        #    covariances_, covariance_type)
        
        return weights_, means_, covariances_