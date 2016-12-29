from pandas import read_csv, get_dummies, to_datetime, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

df = read_csv('enigma-us.gov.dol.oflc.h1b.2014-56e1326ac5eb020a9c4727d1eb8d96e6.csv')

# == Create filters for the analysis ==
#  Applications in NYC (ends up catching one record for Sleepy Hollow, NY) for full time employment
nyc1_filter = df['lca_case_workloc1_city'].str.contains('NEW YORK').fillna(False)
nys1_filter = df['lca_case_workloc1_state'].str.contains('NY').fillna(False)
nyc2_filter = df['lca_case_workloc2_city'].str.contains('NEW YORK').fillna(False)
nys2_filter = df['lca_case_workloc2_state'].str.contains('NY').fillna(False)
nyc_filter = (nyc1_filter & nys1_filter) | (nyc2_filter & nys2_filter)
keep_pw2 = ~(nyc1_filter & nys1_filter) & (nyc2_filter & nys2_filter)

# consolidate prevailing wage information into single columns
df['pw'] = df['pw_1']
df.loc[keep_pw2, 'pw'] = df.loc[keep_pw2, 'pw_2']
df['pw_unit'] = df['pw_unit_1']
df.loc[keep_pw2, 'pw_unit'] = df.loc[keep_pw2, 'pw_unit_2']

fulltime_filter = df['full_time_pos'].eq('Y')
df = df.loc[nyc_filter & fulltime_filter, :].copy()

# include only certified and denied applications
status_filter = df['status'].isin(['CERTIFIED', 'DENIED'])
df = df.loc[status_filter, :].copy()

# clean employer - completely remove spaces - to act as predictors, they don't need to be easily readable
df['cleaned_employer_name'] = df['lca_case_employer_name'].\
    str.replace(r'\(.*?\)', ' ').\
    str.replace(r'''[!"\#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~ ]''', ' ').\
    str.replace(r'((?:^|\s{1,})[A-Z]{1,3}(?:\s{1,}|$))', ' ').\
    str.replace(r'^\s+|\s{2,}|\s+$', ' ').\
    str.replace(r'((?:^|\s{1,})[A-Z]{1,3}(?:\s{1,}|$))', ' ').\
    str.replace(r'^\s+|\s{2,}|\s+$', ' ').\
    str.replace(r'\s+', '')

# If cleaning removed the name entirely, fill in with the original name
df.loc[df['cleaned_employer_name'].eq(''), 'cleaned_employer_name'] = \
    df.loc[df['cleaned_employer_name'].eq(''), 'lca_case_employer_name'].\
    str.replace(r'\(.*?\)', ' ').\
    str.replace(r'''[!"\#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~ ]''', ' ').\
    str.replace(r'\s+', '')

# Convert wages to yearly equivalent
hourly_rate_filter = df['lca_case_wage_rate_unit'].eq('Hour')
monthly_rate_filter = df['lca_case_wage_rate_unit'].eq('Month')
weekly_rate_filter = df['lca_case_wage_rate_unit'].eq('Week')
biweekly_rate_filter = df['lca_case_wage_rate_unit'].eq('Bi-Weekly')
has_upper_wage_filter = df['lca_case_wage_rate_to'].notnull()
df['has_upper_wage_value'] = has_upper_wage_filter

df['average_wage_rate'] = df['lca_case_wage_rate_from'].copy()
df.loc[has_upper_wage_filter, 'average_wage_rate'] = (
    df.loc[has_upper_wage_filter, 'lca_case_wage_rate_from'] +
    df.loc[has_upper_wage_filter, 'lca_case_wage_rate_to']) / 2.

df.loc[hourly_rate_filter, 'average_wage_rate'] *= 2080
df.loc[monthly_rate_filter, 'average_wage_rate'] *= 12
df.loc[weekly_rate_filter, 'average_wage_rate'] *= 52
df.loc[biweekly_rate_filter, 'average_wage_rate'] *= 26

# Convert prevailing wages to yearly equivalent
pw_hourly_rate_filter = df['pw_unit'].eq('Hour')
pw_monthly_rate_filter = df['pw_unit'].eq('Month')
pw_weekly_rate_filter = df['pw_unit'].eq('Week')
pw_biweekly_rate_filter = df['pw_unit'].eq('Bi-Weekly')

df['prevailing_wage'] = df['pw'].copy()
df = df.dropna(subset=['prevailing_wage'])
df.loc[pw_hourly_rate_filter, 'prevailing_wage'] *= 2080
df.loc[pw_monthly_rate_filter, 'prevailing_wage'] *= 12
df.loc[pw_weekly_rate_filter, 'prevailing_wage'] *= 52
df.loc[pw_biweekly_rate_filter, 'prevailing_wage'] *= 26

# Remove outliers
nyc_lower = df['average_wage_rate'].quantile(0.01)
nyc_upper = df['average_wage_rate'].quantile(0.99)
nyc_inliers = df['average_wage_rate'].ge(nyc_lower) & df['average_wage_rate'].le(nyc_upper)
df = df.loc[nyc_inliers, :].copy()

# predictors involving date ranges
df['days_from_submit_to_start'] = \
    (to_datetime(df['lca_case_employment_start_date']) - to_datetime(df['lca_case_submit'])).\
    apply(lambda x: x.days)

df['days_from_start_to_end'] = \
    (to_datetime(df['lca_case_employment_end_date']) - to_datetime(df['lca_case_employment_start_date'])).\
    apply(lambda x: x.days)

# calculate difference from prevailing wage
df['diff_from_prevailing_wage'] = df['average_wage_rate'] - df['prevailing_wage']

# replace any soc names with 10 or fewer instances with "other"
soc_name_counts = df['lca_case_soc_name'].value_counts()
soc_name_replace = soc_name_counts[soc_name_counts.le(10)].index
df.loc[df['lca_case_soc_name'].isin(soc_name_replace), 'lca_case_soc_name'] = 'Other'

# replace any employer names with 10 or fewer instances with "other"
emp_name_counts = df['cleaned_employer_name'].value_counts()
emp_name_replace = emp_name_counts[emp_name_counts.le(10)].index
df.loc[df['cleaned_employer_name'].isin(emp_name_replace), 'cleaned_employer_name'] = 'Other'

# create dummy variables for categorical data
employer_state_dummies = get_dummies(df['lca_case_employer_state'])
soc_name_dummies = get_dummies(df['lca_case_soc_name'])
employer_name_dummies = get_dummies(df['cleaned_employer_name'])

# set aside the outcome variable
certified = df['status'].eq('CERTIFIED').astype('float')

# tie the dummy variables to the rest of the data set

keep_variables = [
    'total_workers', 'diff_from_prevailing_wage', 'prevailing_wage', 'has_upper_wage_value', 'average_wage_rate',
    'days_from_submit_to_start', 'days_from_start_to_end'
]

df = df[keep_variables].\
    merge(employer_state_dummies, left_index=True, right_index=True).\
    merge(employer_name_dummies, left_index=True, right_index=True).\
    merge(soc_name_dummies, left_index=True, right_index=True).\
    astype('float')


# set up the algorithm
rf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_split=1e-07,
    bootstrap=True,
    oob_score=True,
    n_jobs=1,
    random_state=42,
    verbose=0,
    warm_start=False,
    class_weight='balanced_subsample'
)

# fit the model using 5-fold cross-validation
output = DataFrame()
skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
n = 5
for train_index, test_index in skf.split(df, certified):
    print n,
    x_train, x_test = df.iloc[train_index, :], df.iloc[test_index, :]
    y_train, y_test = certified.iloc[train_index], certified.iloc[test_index]

    _ = rf.fit(x_train, y_train)
    y_pred = rf.predict_proba(x_test)

    y_test = y_test.to_frame('original')
    y_test['prediction'] = y_pred[:, 1]

    output = output.append(y_test.copy(), ignore_index=True)
    n -= 1

# evalute false postive/negative rate under different cutoff assumptions
evaluation = DataFrame(index=[n / 100. for n in range(10, 100, 10)], columns=['false_positives', 'false_negatives'])
for i in evaluation.index:
    output['prediction_cut'] = output['prediction'].gt(i).astype('float')
    summary = output.groupby('original')['prediction_cut'].value_counts() / output.shape[0]
    evaluation.loc[i, 'false_negatives'] = summary.loc[1.0, 0.0]
    evaluation.loc[i, 'false_positives'] = summary.loc[0.0, 1.0]
evaluation = evaluation.astype('float')

# put the importance scores in a human-readable format
importances = DataFrame({'scores': rf.feature_importances_}, index=df.columns).\
    sort_values('scores', ascending=False)

