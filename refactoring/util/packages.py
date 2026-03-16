import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime as dt
import re
from sklearn.impute import SimpleImputer
from pandas.api.types import (
    is_datetime64_any_dtype, is_timedelta64_dtype,
    is_object_dtype, is_numeric_dtype
)
import datetime as dt
from pandas.api.types import (
    is_datetime64_any_dtype, is_timedelta64_dtype,
    is_object_dtype, is_numeric_dtype
)

import random


from scipy.special import logsumexp
from scipy.stats import wilcoxon, spearmanr
from stepmix.stepmix import StepMix
from sklearn.model_selection import train_test_split
from itertools import zip_longest
import string

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager as fm
from matplotlib import rcParams


from types import SimpleNamespace



# KMO, Bartlett's test, EFA 등
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo

from semopy import Model, calc_stats


# Shapiro–Wilk 정규성 검정
from scipy.stats import shapiro




# from __future__ import annotations

from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score


from scipy.stats import pearsonr



from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, 
                             f1_score, roc_auc_score, make_scorer)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier