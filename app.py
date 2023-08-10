#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from htest import t_test_one_sample, t_test_two_sample, plot_graph

flights = pd.read_csv('flights.csv', index_col=False)

"""
My friend flew between New York and California and
claims that it takes only 5 hours at the most (300 minutes)

However, we think that it actually takes longer

H_0: mu = 300
H_a: mu > 300
"""
"""
ny_to_ca = flights[(flights['ORIGIN_STATE_NM'] == 'New York') & (flights['DEST_STATE_NM'] == 'California')]
ny_to_ca_sample = ny_to_ca['AIR_TIME'].sample(20, random_state=7)

mu = 300
alpha = 0.05

t_score, p_value = t_test_one_sample(mu, ny_to_ca_sample, 'upper')
if p_value < alpha:
    print('p={}\nReject H_0 and accept H_a'.format(p_value))
else:
    print('p={}\nNot enough evidence to reject H_0'.format(p_value))

plot_graph(t_score, 19, 'upper', alpha)
"""

"""
Suppose we want to determine how to allocation funds to different
airports based on the average flight's distance.
The two airports we're considering is New York City and Niagara Falls

H_0: mu1 - mu2 = 0
H_a: mu1 > mu2    ->    mu1 - mu2 > 0
"""

nyc_flights = flights[flights['ORIGIN_CITY_NAME'] == 'New York, NY']
niagara_falls_flights = flights[flights['ORIGIN_CITY_NAME'] == 'Niagara Falls, NY']

nyc_sample = nyc_flights['DISTANCE'].sample(20, random_state=17)
niagara_falls_sample = niagara_falls_flights['DISTANCE'].sample(20, random_state=17)

alpha = 0.05

t_score, p_value = t_test_two_sample(nyc_sample, niagara_falls_sample, 'upper')
if p_value < alpha:
    print('p={}\nReject H_0 and accept H_a'.format(p_value))
else:
    print('p={}\nNot enough evidence to reject H_0'.format(p_value))

plot_graph(t_score, 38, 'upper', alpha)
