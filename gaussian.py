#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from scipy import stats

# heights of adult males ~ N(70, 4)
mean = 70
std = 4
heights = stats.norm(mean, std)

# compute p(X < 70)
print(heights.cdf(70))
# compute p(X > 70)
print(1 - heights.cdf(70))
print(heights.sf(70))

# compute area between += 1 std (~68%)
print(heights.cdf(70+4) - heights.cdf(70-4))

# compute area between += 2 std (~95%)
print(heights.cdf(70+4*2) - heights.cdf(70-4*2))

# compute area between += 3 std (~99.7%)
print(heights.cdf(70+4*3) - heights.cdf(70-4*3))
