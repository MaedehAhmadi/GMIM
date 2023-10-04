import numpy as np 
from scipy import linalg
import torch

device = torch.device("cuda:0")

def _asarray_validated(a, check_finite=True,
                       sparse_ok=False, objects_ok=False, mask_ok=False,
                       as_inexact=False):
    """
    Helper function for SciPy argument validation.
    Many SciPy linear algebra functions do support arbitrary array-like
    input arguments. Examples of commonly unsupported inputs include
    matrices containing inf/nan, sparse matrix representations, and
    matrices with complicated elements.
    Parameters
    ----------
    a : array_like
        The array-like input.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
        Default: True
    sparse_ok : bool, optional
        True if scipy sparse matrices are allowed.
    objects_ok : bool, optional
        True if arrays with dype('O') are allowed.
    mask_ok : bool, optional
        True if masked arrays are allowed.
    as_inexact : bool, optional
        True to convert the input array to a np.inexact dtype.
    Returns
    -------
    ret : ndarray
        The converted validated array.
    """
    if not sparse_ok:
        import scipy.sparse
        if scipy.sparse.issparse(a):
            msg = ('Sparse matrices are not supported by this function. '
                   'Perhaps one of the scipy.sparse.linalg functions '
                   'would work instead.')
            raise ValueError(msg)
    if not mask_ok:
        if np.ma.isMaskedArray(a):
            raise ValueError('masked arrays are not supported')
    toarray = np.asarray_chkfinite if check_finite else np.asarray
    a = toarray(a)
    if not objects_ok:
        if a.dtype is np.dtype('O'):
            raise ValueError('object arrays are not supported')
    if as_inexact:
        if not np.issubdtype(a.dtype, np.inexact):
            a = toarray(a, dtype=np.float_)
    return a
    
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
    """Compute the log of the sum of exponentials of input elements.
    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed.
        .. versionadded:: 0.11.0
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the original array.
        .. versionadded:: 0.15.0
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`. These values may be negative in order to
        implement subtraction.
        .. versionadded:: 0.12.0
    return_sign : bool, optional
        If this is set to True, the result will be a pair containing sign
        information; if False, results that are negative will be returned
        as NaN. Default is False (no sign information).
        .. versionadded:: 0.16.0
    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.
    sgn : ndarray
        If return_sign is True, this will be an array of floating-point
        numbers matching res and +1, 0, or -1 depending on the sign
        of the result. If False, only one result is returned.
    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2
    Notes
    -----
    NumPy has a logaddexp function which is very similar to `logsumexp`, but
    only handles two arguments. `logaddexp.reduce` is similar to this
    function, but may be less stable.
    Examples
    --------
    >>> from scipy.special import logsumexp
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    With weights
    >>> a = np.arange(10)
    >>> b = np.arange(10, 0, -1)
    >>> logsumexp(a, b=b)
    9.9170178533034665
    >>> np.log(np.sum(b*np.exp(a)))
    9.9170178533034647
    Returning a sign flag
    >>> logsumexp([1,2],b=[1,-1],return_sign=True)
    (1.5413248546129181, -1.0)
    Notice that `logsumexp` does not directly support masked arrays. To use it
    on a masked array, convert the mask into zero weights:
    >>> a = np.ma.array([np.log(2), 2, np.log(3)],
    ...                  mask=[False, True, False])
    >>> b = (~a.mask).astype(int)
    >>> logsumexp(a.data, b=b), np.log(5)
    1.6094379124341005, 1.6094379124341005
    """
    #a = _asarray_validated(a, check_finite=False)
    if b is not None:
        a, b = np.broadcast_arrays(a, b)
        if np.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -np.inf
 
    a_max, _ = torch.max(a, keepdim=True, dim=axis)

    if a_max.dim() > 0:
        a_max[torch.isinf(a_max)] = 0
    elif not torch.isinf(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        tmp = b * np.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    # suppress warnings about log of zero
    #with np.errstate(divide='ignore'):
    s = torch.sum(tmp, dim=axis, keepdim=keepdims)
    if return_sign:
        sgn = torch.sign(s)
        s *= sgn  # /= makes more sense but we need zero -> zero
    out = torch.log(s)

    if not keepdims:
        a_max = torch.squeeze(a_max, dim=axis)
    out += a_max

    if return_sign:
        return out, sgn
    else:
        return out


def _compute_precision_cholesky(covariances, covariance_type):
    """Compute the Cholesky decomposition of the precisions.
    Parameters
    ----------
    covariances : array-like
        The covariance matrix of the current components.
        The shape depends of the covariance_type.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of precision matrices.
    Returns
    -------
    precisions_cholesky : array-like
        The cholesky decomposition of sample precisions of the current
        components. The shape depends of the covariance_type.
    """
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.")

    if covariance_type == 'full':
        print("shhhhhhhhhhhh  ", covariances.shape)
        n_components, n_features, _ = covariances.shape
        precisions_chol = np.empty((n_components, n_features, n_features))
        for k, covariance in enumerate(covariances):
            try:
                cov_chol = linalg.cholesky(covariance, lower=True)
                #cov_chol = torch.linalg.cholesky_ex(covariances, upper=False)
            except linalg.LinAlgError:
                raise ValueError(estimate_precision_error_message)
            precisions_chol[k] = linalg.solve_triangular(cov_chol,
                                                         np.eye(n_features),
                                                         lower=True).T
            precisions_chol = torch.linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  upper=False).T                                                         
    elif covariance_type == 'tied':
        _, n_features = covariances.shape
        try:
            cov_chol = linalg.cholesky(covariances, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol = linalg.solve_triangular(cov_chol, np.eye(n_features),
                                                  lower=True).T
    else:
        #if np.any(np.less_equal(covariances, 0.0)):
        #   raise ValueError(estimate_precision_error_message)
        precisions_chol = 1. / torch.sqrt(covariances)
    return precisions_chol
    
    
#self.precisions_cholesky_ = _compute_precision_cholesky(
#                covariances, covariance_type)

def _estimate_log_weights(weights):
    return torch.log(weights)

'''               
def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == 'dirichlet_process':
            digamma_sum = digamma(self.weight_concentration_[0] +
                                  self.weight_concentration_[1])
            digamma_a = digamma(self.weight_concentration_[0])
            digamma_b = digamma(self.weight_concentration_[1])
            return (digamma_a - digamma_sum +
                    np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1])))
        else:
            # case Variationnal Gaussian mixture with dirichlet distribution
            return (digamma(self.weight_concentration_) -
                    digamma(np.sum(self.weight_concentration_)))
'''
def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (torch.sum(torch.log(matrix_chol), dim=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol

def _estimate_log_gaussian_prob(X, means, covariances, covariance_type, precisions_chol):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    means : array-like of shape (n_components, n_features)
    precisions_chol : array-like
        Cholesky decompositions of the precision matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)
    #for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
    #        print("shhhhhh   ", X.shape, prec_chol.shape, mu.shape)
    
    if covariance_type == 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'tied':
        log_prob = np.empty((n_samples, n_components))
        for k, mu in enumerate(means):
            y = np.dot(X, precisions_chol) - np.dot(mu, precisions_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type == 'diag':
        
        precisions = precisions_chol ** 2
        
        log_prob = (torch.sum((means ** 2 * precisions), 1) -
                    2. * torch.matmul(X, (means * precisions).t()) +
                    torch.matmul(X ** 2, precisions.t()))

    elif covariance_type == 'spherical':
        precisions = precisions_chol ** 2
        log_prob = (np.sum(means ** 2, 1) * precisions -
                    2 * np.dot(X, means.T * precisions) +
                    np.outer(row_norms(X, squared=True), precisions))
    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def _estimate_log_prob(X, means, covariances, covariance_type):
        precisions_chol = _compute_precision_cholesky(
                covariances, covariance_type)#.to(device)
        return _estimate_log_gaussian_prob(
            X, means, covariances, covariance_type, precisions_chol)
'''
def _estimate_log_prob(self, X):
        _, n_features = X.shape
        # We remove `n_features * np.log(self.degrees_of_freedom_)` because
        # the precision matrix is normalized
        log_gauss = (_estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type) -
            .5 * n_features * np.log(self.degrees_of_freedom_))

        log_lambda = n_features * np.log(2.) + np.sum(digamma(
            .5 * (self.degrees_of_freedom_ -
                  np.arange(0, n_features)[:, np.newaxis])), 0)

        return log_gauss + .5 * (log_lambda -
                                 n_features / self.mean_precision_)
'''
                                 
def _estimate_weighted_log_prob(X, means, covariances, covariance_type, weights):
        """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        weighted_log_prob : array, shape (n_samples, n_component)
        """
        return _estimate_log_prob(X, means, covariances, covariance_type) + _estimate_log_weights(weights)

def estimate_log_prob_resp(X, means, covariances, covariance_type, weights):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = _estimate_weighted_log_prob(X, means, covariances, covariance_type, weights)

        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        #print("iiii 111  " , weighted_log_prob)
        #print("iiii  2222" , log_prob_norm)
        #with np.errstate(under='ignore'):
            # ignore underflow
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        #print("iiii 3333 ", log_resp)   
        return log_prob_norm, log_resp
        
def predict_proba(X, means, covariances, weights, covariance_type):
    """Predict posterior probability of each component given the data.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.
    Returns
    -------
    resp : array, shape (n_samples, n_components)
        Returns the probability each Gaussian (state) in
        the model given each sample.
    """
    #X = _check_X(X, None, means_.shape[1])
    _, log_resp = estimate_log_prob_resp(X, means, covariances, covariance_type, weights)
    return torch.exp(log_resp)