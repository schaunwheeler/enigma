from pandas import read_csv
from numpy import mean, log
from numpy.random import choice
from bokeh.charts import Scatter, output_file, show

df = read_csv('enigma-us.gov.dol.oflc.h1b.2014-56e1326ac5eb020a9c4727d1eb8d96e6.csv')

# == Clean employer names ==
#  still problems of possible extraneous words ("LIMITED"), and typing errors (accidental spaces, etc.)

df['cleaned_employer_name'] = df['lca_case_employer_name'].\
    str.replace(r'\(.*?\)', ' ').\
    str.replace(r'''[!"\#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~ ]''', ' ').\
    str.replace(r'((?:^|\s{1,})[A-Z]{1,3}(?:\s{1,}|$))', ' ').\
    str.replace(r'^\s+|\s{2,}|\s+$', ' ').\
    str.replace(r'((?:^|\s{1,})[A-Z]{1,3}(?:\s{1,}|$))', ' ').\
    str.replace(r'^\s+|\s{2,}|\s+$', ' ').\
    str.strip()

# If cleaning removed the name entirely, fill in with the original name
df.loc[df['cleaned_employer_name'].eq(''), 'cleaned_employer_name'] = \
    df.loc[df['cleaned_employer_name'].eq(''), 'lca_case_employer_name']

# == Create filters for the analysis ==
#  Applications in NYC (ends up catching one record for Sleepy Hollow, NY)
nyc1_filter = df['lca_case_workloc1_city'].str.contains('NEW YORK').fillna(False)
nys1_filter = df['lca_case_workloc1_state'].str.contains('NY').fillna(False)
nyc2_filter = df['lca_case_workloc2_city'].str.contains('NEW YORK').fillna(False)
nys2_filter = df['lca_case_workloc2_state'].str.contains('NY').fillna(False)
nyc_filter = (nyc1_filter & nys1_filter) | (nyc2_filter & nys2_filter)

#  Applications in Mountain View (might have missed a few alternate spellings)
mtv1_filter = df['lca_case_workloc1_city'].str.contains('MOU?N?a?TAI?N\s*VIEW').fillna(False)
ca1_filter = df['lca_case_workloc1_state'].str.contains('CA').fillna(False)
mtv2_filter = df['lca_case_workloc2_city'].str.contains('MOU?N?a?TAI?N\s*VIEW').fillna(False)
ca2_filter = df['lca_case_workloc2_state'].str.contains('CA').fillna(False)
mtv_filter = (mtv1_filter & ca1_filter) | (mtv2_filter & ca2_filter)

#  Applications for full-tie workers (part-time workers made up only 2.7% of all the applications)
fulltime_filter = df['full_time_pos'].eq('Y')

# Applications with different wage rate units
#  92% of wages are yearly, 7% hourly, and less than 1% monthly, weekly, or bi-weekly
hourly_rate_filter = df['lca_case_wage_rate_unit'].eq('Hour')
monthly_rate_filter = df['lca_case_wage_rate_unit'].eq('Month')
weekly_rate_filter = df['lca_case_wage_rate_unit'].eq('Week')
biweekly_rate_filter = df['lca_case_wage_rate_unit'].eq('Bi-Weekly')

# Applications that have an upper range listed to the proposed wages
has_upper_wage_filter = df['lca_case_wage_rate_to'].notnull()


# ==Standardize definitions witin the dataset==

#  Calculate a single wage for each application
df['average_wage_rate'] = df['lca_case_wage_rate_from']
df.loc[has_upper_wage_filter, 'average_wage_rate'] = (
    df.loc[has_upper_wage_filter, 'lca_case_wage_rate_from'] +
    df.loc[has_upper_wage_filter, 'lca_case_wage_rate_to']) / 2.

#  Calculate yearly wage
#  assumes an 8-hour work day, 5-day work week, and 52-week work year
df.loc[hourly_rate_filter, 'average_wage_rate'] *= 2080
df.loc[monthly_rate_filter, 'average_wage_rate'] *= 12
df.loc[weekly_rate_filter, 'average_wage_rate'] *= 52
df.loc[biweekly_rate_filter, 'average_wage_rate'] *= 26

# separate out NYC and Mountain View datasets
df_nyc = df.loc[nyc_filter & fulltime_filter, :].copy()
df_mtv = df.loc[mtv_filter & fulltime_filter, :].copy()

# Flag outliers (bottom 1% and top 1%)
# Distribution has a long enough tail that boxplots are a bad way to identify outliers
nyc_lower = df_nyc['average_wage_rate'].quantile(0.01)
nyc_upper = df_nyc['average_wage_rate'].quantile(0.99)
nyc_inliers = df_nyc['average_wage_rate'].ge(nyc_lower) & df_nyc['average_wage_rate'].le(nyc_upper)

mtv_lower = df_mtv['average_wage_rate'].quantile(0.01)
mtv_upper = df_mtv['average_wage_rate'].quantile(0.99)
mtv_inliers = df_mtv['average_wage_rate'].ge(mtv_lower) & df_mtv['average_wage_rate'].le(mtv_upper)

# NYC had applications filed for 61216 workers, Mountain view for 11669
df_nyc_inliers = df_nyc.loc[nyc_inliers, :].dropna(subset=['average_wage_rate', 'total_workers']).copy()
df_mtv_inliers = df_mtv.loc[mtv_inliers, :].dropna(subset=['average_wage_rate', 'total_workers']).copy()

# Simulate means and standard deviations, and calculate probability of higher wages in each city
#  Mountain view data appears normal: mean of 120213, median of 120245
#  NYC data appears *not* normal: mean of 89064, median of 79627

nyc_means = []
mtv_means = []
nyc_stdevs = []
mtv_stdevs = []
prob_mtv_gt_nyc_wage = 0
prob_nyc_gt_mtv_wage = 0

for i in range(1000):
    nyc_sample = df_nyc_inliers['average_wage_rate'].\
        sample(n=10000, replace=True, weights=df_nyc_inliers['total_workers'])
    mtv_sample = df_mtv_inliers['average_wage_rate'].\
        sample(n=10000, replace=True, weights=df_mtv_inliers['total_workers'])
    nyc_means.append(nyc_sample.mean())
    mtv_means.append(mtv_sample.mean())
    nyc_stdevs.append(nyc_sample.std())
    mtv_stdevs.append(mtv_sample.std())
    prob_mtv_gt_nyc_wage += (mtv_sample.values > nyc_sample.values).mean()
    prob_nyc_gt_mtv_wage += (nyc_sample.values > mtv_sample.values).mean()
prob_mtv_gt_nyc_wage /= 1000.
prob_nyc_gt_mtv_wage /= 1000.

nyc_wage_mean = mean(nyc_means)
nyc_wage_stdev = mean(nyc_stdevs)

mtv_wage_mean = mean(mtv_means)
mtv_wage_stdev = mean(mtv_stdevs)

prob_mtv_gt_nyc_meanwage = 0
for i in range(1000):
    mtv_sim = choice(mtv_means, size=len(mtv_means), replace=False)
    nyc_sim = choice(nyc_means, size=len(nyc_means), replace=False)

    prob_mtv_gt_nyc_meanwage += (mtv_sim > nyc_sim).mean()
prob_mtv_gt_nyc_meanwage /= 1000.

# Calculate number of employer applciations and average proposed wages
employer_counts = df_nyc.groupby('cleaned_employer_name')['total_workers'].sum()
employer_wages = df_nyc_inliers.groupby('cleaned_employer_name')['average_wage_rate'].mean()

employer_counts_and_wages = employer_counts.to_frame('counts').merge(
    employer_wages.to_frame('wages'),
    left_index=True, right_index=True)
employer_counts_and_wages['logged_counts'] = log(employer_counts_and_wages['counts'])
employer_counts_and_wages = employer_counts_and_wages.sort_values('counts', ascending=False)

# Plot number of applications vs. average wages
p = Scatter(
    employer_counts_and_wages,
    x='logged_counts', y='wages',
    xlabel="Number of applications (log scale)", ylabel="Average wages",
    title="HB1 Visas applied for vs average wages in NYC"
)
output_file("applications_vs_wages_graph.html")
show(p)
