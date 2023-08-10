#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from scipy import stats

STANDARD_NORM = stats.norm(0, 1)

"""
# Determine our hypotheses
# H_0: mu = 70
# H_a: mu > 70
alpha = 0.05

mean = 70
std = 4
n = 40
x = 72

# Compute test statistic
z = (x - mean) / (std / math.sqrt(n))

# Lookup p-value
# sf(z) = 1 - cdf(z)
p_value = STANDARD_NORM.sf(z)

# Decide on our hypotheses
if p_value < alpha:
    print('Reject H_0 and accept H_a: (z={}, p={})'.format(z, p_value))
else:
    print('Not enough evidence to reject H_0: (z={}, p={})'.format(z, p_value))
"""

"""
# Determine our hypotheses
# H_0: mu = 82
# H_a: mu != 82
alpha = 0.01

mean = 82
std = 5
n = 60
x = 84

# Compute test statistic
z = (x - mean) / (std / math.sqrt(n))

# Lookup p-value
p_value = 2 * STANDARD_NORM.sf(abs(z))

# Decide on our hypotheses
if p_value < alpha:
    print('Reject H_0 and accept H_a: (z={}, p={})'.format(z, p_value))
else:
    print('Not enough evidence to reject H_0: (z={}, p={})'.format(z, p_value))
"""

"""
# Determine our hypotheses
# H_0: mu_1 - mu_2 = 0
# H_a: mu_1 - mu_2 != 0
alpha = 0.05

n_control = 1000
x_control = 185
s_control = 39

n_experimental = 900
x_experimental = 180
s_experimental = 50

# Compute test statistic
z = (x_control - x_experimental) / math.sqrt(s_control ** 2 / n_control + s_experimental ** 2 / n_experimental)

# Lookup p-value
p_value = 2 * STANDARD_NORM.sf(abs(z))

# Decide on our hypotheses
if p_value < alpha:
    print('Reject H_0 and accept H_a: (z={}, p={})'.format(z, p_value))
else:
    print('Not enough evidence to reject H_0: (z={}, p={})'.format(z, p_value))
"""

"""
# Determine our hypotheses
# H_0: mu_1 = 70
# H_a: mu_1 > 70
alpha = 0.05

mu = 70
sample_mean = 72
sample_std = 5
n = 20

# construct t-distribution
dof = n - 1
t_dist = stats.t(dof)

# Compute test statistic
t = (sample_mean - mu) / (sample_std / math.sqrt(n))

# Lookup p-value
t_critical = t_dist.ppf(1 - alpha)

# Decide on our hypotheses
if t > t_critical:
    print('Reject H_0 and accept H_a: (t={}, t_critical={})'.format(t, t_critical))
else:
    print('Not enough evidence to reject H_0: (t={}, t_critical={})'.format(t, t_critical))

p_value = t_dist.sf(t)
if p_value < alpha:
    print('Reject H_0 and accept H_a: (t={}, p={})'.format(t, p_value))
else:
    print('Not enough evidence to reject H_0: (t={}, p={})'.format(t, p_value))
"""

# Determine our hypotheses
# H_0: mu_1 - mu_2 = 0
# H_a: mu_1 - mu_2 != 0
alpha = 0.05

sample_mean1 = 70
sample_std1 = 5
n1 = 30

sample_mean2 = 72
sample_std2 = 4
n2 = 20

# construct t-distribution
dof = n1 + n2 - 2
t_dist = stats.t(dof)

# Compute test statistic
s_p = math.sqrt(((n1 - 1) * sample_std1 ** 2 + (n2 - 1) * sample_std2 ** 2) / dof)
t = (sample_mean1 - sample_mean2) / (s_p * math.sqrt(1. / n1 + 1. / n2))

# Lookup p-value
p_value = 2 * t_dist.sf(abs(t))

# Decide on our hypotheses
if p_value < alpha:
    print('Reject H_0 and accept H_a: (t={}, p={})'.format(t, p_value))
else:
    print('Not enough evidence to reject H_0: (t={}, p={})'.format(t, p_value))
