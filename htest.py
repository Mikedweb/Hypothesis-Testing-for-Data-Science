import math
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

def _compute_p_value(t_dist, t, test_type):
    if test_type == 'lower':
        p_value = t_dist.cdf(t)
    elif test_type == 'upper':
        p_value = t_dist.sf(t)
    elif test_type == 'two-tailed':
        p_value = 2 * t_dist.sf(abs(t))
    else:
        raise Exception('Unknown test type: {}'.format(test_type))
    return p_value

def t_test_one_sample(mu, data, test_type):
    """Computes a one-sample t-test against our data.
    
    Args:
        mu (float): Null hypothesis mu assumption.
        data (ndarray): Raw sample data.
        test_type ({'lower', 'upper', 'two-tailed'}): Type of test.
    
    Raises:
        Exception: test_type must be a valid test.
    
    Returns:
        (t, p-value): Returns t-score and corresponding p-value
    """
    # compute mean, standard deviation, and degrees of freedom
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    n = len(data)
    dof = n - 1
    
    # generate the corresponding t-distribution
    t_dist = stats.t(dof)
    
    # compute t-score
    t = (sample_mean - mu) / (sample_std / math.sqrt(n))
    
    # compute p-value
    p_value = _compute_p_value(t_dist, t, test_type)
    return t, p_value

def t_test_two_sample(sample1, sample2, test_type):
    """Computes a two-sample t-test against our data.
    
    Args:
        sample1 (ndarray): first sample
        sample2 (ndarray): second sample
        test_type ({'lower', 'upper', 'two-tailed'}): Type of test.
    
    Raises:
        Exception: test_type must be a valid test.
    
    Returns:
        (t, p-value): Returns t-score and corresponding p-value
    """
    # compute mean, standard deviation, and degrees of freedom
    sample_mean1 = np.mean(sample1)
    sample_mean2 = np.mean(sample2)
    sample_std1 = np.std(sample1, ddof=1)
    sample_std2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)
    dof = n1 + n2 - 2
    
    # generate corresponding t-distribution
    t_dist = stats.t(dof)
    
    # compute pooled standard deviation
    s_p = math.sqrt(((n1 - 1) * sample_std1 ** 2 + (n2 - 1) * sample_std2 ** 2) / dof)
    
    # compute t-score
    t = (sample_mean1 - sample_mean2) / (s_p * math.sqrt(1. / n1 + 1. / n2))
    
    # compute p-value
    p_value = _compute_p_value(t_dist, t, test_type)
    
    return t, p_value

def plot_graph(t, dof, test_type, critical_value):
    """Plots the t-distribution with dof degrees of freedom and highlights p-value and rejection region
    
    Args:
        t (float): t-score
        dof (int): degrees of freedom for t-distribution
        test_type ({'lower', 'upper', 'two-tailed'}): type of test
        critical_value (float): alpha level
    
    Raises:
        Exception: test_type must be a valid test.
    """
    MIN_PPF = 0.001
    MAX_PPF = 0.999

    # construct the t-dist
    t_dist = stats.t(dof)
    x = np.linspace(t_dist.ppf(MIN_PPF), t_dist.ppf(MAX_PPF), 100)
    y = t_dist.pdf(x)
    plt.plot(x, y, 'b')

    # show p-value
    if test_type == 'upper':
        # beyond our graph
        if t > x.max():
            return

        # plot p-value
        plt.fill_between(x[x > t], 0, y[x > t], facecolor='b', alpha=0.5, label='p-value')

        # plot rejection region
        x_alpha = np.linspace(t_dist.ppf(1 - critical_value), t_dist.ppf(MAX_PPF), 100)
        y_alpha = t_dist.pdf(x_alpha)
        plt.fill_between(x_alpha, 0, y_alpha, facecolor='r', alpha=0.5, label='rejection region')

    elif test_type == 'lower':
        # beyond our graph
        if t < x.min():
            return

        # plot p-value
        plt.fill_between(x[x < t], 0, y[x < t], facecolor='b', alpha=0.5, label='p-value')

        # plot rejection region
        x_alpha = np.linspace(t_dist.ppf(MIN_PPF), t_dist.ppf(critical_value), 100)
        y_alpha = t_dist.pdf(x_alpha)
        plt.fill_between(x_alpha, 0, y_alpha, facecolor='r', alpha=0.5, label='rejection region')

    elif test_type == 'two-tailed':
        # beyond our graph
        if t > x.max() or t < x.min():
            return

        # plot p-value
        plt.fill_between(x[x > abs(t)], 0, y[x > abs(t)], facecolor='b', alpha=0.5, label='p-value')
        plt.fill_between(x[x < -abs(t)], 0, y[x < -abs(t)], facecolor='b', alpha=0.5)

        # plot rejection region
        x_alpha = np.linspace(t_dist.ppf(MIN_PPF), t_dist.ppf(critical_value / 2), 100)
        y_alpha = t_dist.pdf(x_alpha)
        plt.fill_between(x_alpha, 0, y_alpha, facecolor='r', alpha=0.5, label='rejection region')
        plt.fill_between(abs(x_alpha), 0, y_alpha, facecolor='r', alpha=0.5)

    else:
        raise Exception('Unknown test type: {}'.format(test_type))
    
    plt.legend()
    plt.title('{} t-test ({} DOF) of t={} @ alpha={}'.format(test_type, dof, t, critical_value))

    plt.show()
