import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import scipy.stats as stats
from functools import partial
from pandas.core.dtypes.common import is_numeric_dtype, is_datetime64_dtype


def eda(dataframe, head=True, info=True, describe=True, duplicated=True,
        sort_duplicates=False):
    """
    exploratory data analysis
    :param dataframe: a pandas DataFrame
    :param head: boolean; if True, the first 5 rows of dataframe is shown
    :param info: boolean; if True, dataframe.info() is shown
    :param describe: boolean; if True, descriptions of the columns (grouped by
                     numeric, datetime, and other) are shown
    :param duplicated: boolean; if True, the number and content (if any) of
                       the duplicated rows are shown
    :param sort_duplicates: boolean; if True, the duplicated rows are sorted by
                            each column of the dataframe
    :return: None
    """

    if head:
        print('Head of the dataframe:\n\n{}\n\n'.format(dataframe.head()))

    # shape, index, columns, missing values (infer null from non-null), dtypes
    if info:
        dataframe.info()
        print('\n')

    if describe:
        numeric, datetime, other = False, False, False
        for column in dataframe.columns:
            if is_numeric_dtype(dataframe[column]):
                numeric = True
            elif is_datetime64_dtype(dataframe[column]):
                datetime = True
            else:
                other = True

        # describe numeric columns (if any)
        if numeric:
            print(dataframe.describe())
            print('\n')

        # describe datetime columns (if any)
        if datetime:
            print(dataframe.describe(include=['datetime']))
            print('\n')

        # describe other columns (if any)
        if other:
            print(dataframe.describe(exclude=[np.number, np.datetime64]))
            print('\n')

    # find duplicates
    if duplicated:
        n_duplicates = dataframe.duplicated().sum()
        print('Number of duplicated rows: {}'.format(n_duplicates))
        if n_duplicates > 0:
            print('\n')
            duplicated_df = dataframe[dataframe.duplicated(keep=False)]
            if not sort_duplicates:
                print(duplicated_df)
            else:
                print(duplicated_df.sort_values(list(duplicated_df.columns)))


def clean_column_names(dataframe, inplace=False):
    """
    :param dataframe: a pandas DataFrame
    :return: a new DataFrame with the clean (dot-friendly) column names, i.e.,
    leading and trailing whitespace removed, all letters changed to lowercase,
    characters that are not a letter or number replaced by an underscore (or
    removed if at the end), an underscore added to the beginning if the first
    character is a number
    """
    cleaned_columns = []
    for column_name in dataframe.columns:
        column_name = column_name.strip()
        column_name = column_name.lower()
        column_name = re.sub(r'[^a-z0-9]', '_', column_name)
        column_name = '_' + column_name if re.match(r'\d', column_name) \
                                        else column_name
        column_name = re.sub(r'_+', '_', column_name)
        column_name = column_name.rstrip('_')
        cleaned_columns.append(column_name)

    if not inplace:
        dataframe = dataframe.copy()
    dataframe.columns = cleaned_columns
    return dataframe


def memory_change(input_df, column, dtype):
    df = input_df.copy()
    old = round(df[column].memory_usage(deep=True) / 1024, 2)  # In KB
    new = round(df[column].astype(dtype).memory_usage(deep=True) / 1024, 2)
    change = round(100 * (old - new) / (old), 2)
    print("The initial memory footprint for {column} is: {old}KB.\n"
          "The casted {column} now takes: {new}KB.\n"
          "A change of {change} %.").format(**locals())


def distplot_with_norm_fit(array, bins=10, xlabel=None, ylabel=None, title=None):
    """
    :param array: a np.array or pd.Series
    :param bins: bins argument to feed into sns.distplot,
                 e.g. integer or bin edges
    :param xlabel: label for x-axis
    :param ylabel: label for y-axis
    :param title: title for the plot
    For an array, plot 1) histogram, 2) KDE curve, and 3) normal distribution
    curve with the same mean and standard deviation as the array
    :return: ax object
    """
    norm_fit_plot = partial(sns.distplot, fit=stats.norm,
                            kde_kws={"label": "KDE"},
                            fit_kws={"label": "normal",
                                     "color": "red",
                                     "alpha": 0.7,
                                     "linestyle": "dashed"})

    ax = norm_fit_plot(array, bins=bins)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)
    ax.legend(loc='best')
    return ax


def distplot_with_norm_fit_in_dev(array, **kwargs):
    """
    :param array: a np.array or pd.Series
    :param kwargs: any keyword arguments for sns.distplot
    For an array, plot 1) histogram, 2) KDE curve, and 3) normal distribution
    curve with the same mean and standard deviation as the array
    :return: ax object
    """
    norm_fit_plot = partial(sns.distplot, fit=stats.norm,
                            kde_kws={"label": "KDE"},
                            fit_kws={"label": "normal",
                                     "color": "red",
                                     "alpha": 0.7,
                                     "linestyle": "dashed"})

    ax = norm_fit_plot(array, **kwargs)
    ax.legend(loc='best')
    return ax


def scatter_plot_with_linear_fit(x, y, slope=None, y_intercept=None):
    """
    :param x: an array
    :param y: an array
    :param slope: slope of the fitted line
    :param y_intercept: y-intercept of the fitted line
    If slope or y_intercept is not specified, these parameters will be generated
    by linear fit.
    :return: Pearson correlation coefficient and p-value
    """
    plt.scatter(x, y, alpha=0.8)

    if slope is None or y_intercept is None:
        slope, y_intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(np.min(x), np.max(x), 100)
    y_fit = slope * x_fit + y_intercept
    plt.plot(x_fit, y_fit, linestyle='dashed', color='black', alpha=0.5)

    return stats.pearsonr(x, y)


def scatter_matrix_with_corr(dataframe, plot=True):
    """
    :param dataframe: a pandas dataframe
    :param plot: a Boolean; if True, plot the scatter matrix of all numeric
                 columns in the dataframe
    :return: a dictionary where the keys are tuples of two column names,
             and the values are the Pearson's correlation coefficients
             and the p-values of the two columns
    """
    if plot:
        plt.rcParams['figure.figsize'] = 15, 15
        pd.plotting.scatter_matrix(dataframe)
        # reset figsize
        plt.rcParams['figure.figsize'] = 6, 4

    # create working dataframe with non-numeric columns removed
    working_df = dataframe.copy()
    for column in dataframe.columns:
        if is_numeric_dtype(dataframe[column]):
            working_df[column] = dataframe[column].astype(float)
        else:
            working_df.drop(column, axis=1, inplace=True)

    columns = working_df.columns
    return {(col_x, col_y): stats.pearsonr(working_df[col_x], working_df[col_y])
            for idx, col_x in enumerate(columns)
            for col_y in columns[idx+1:]}


def significant_corr(corr_dict, p=0.05):
    """
    :param corr_dict: dictionary returned by scatter_matrix_with_corr
    :param p: cut-off alpha
    :return: a dictionary where the keys are pairs of numeric variables that are
             statistically significantly correlated, and the values are the
             corresponding Pearson's correlation coefficients
    """
    return {column_pair: corr
            for column_pair, (corr, p_value) in corr_dict.items()
            if p_value < p}


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""

    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1) / n

    return x, y


