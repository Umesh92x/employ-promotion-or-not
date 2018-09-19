'''                               VISUALISATION                        '''
#                     VISAUALISATION BETWEEN CATEGORICAL DATA AND QUANTITITY DATA
X['department'].value_counts().plot(kind='bar')
X['education'].value_counts().plot(kind='bar')
X['gender'].value_counts().plot(kind='bar')
y.value_counts().plot(kind='bar',title='O is far greater then 1')

department_vs_promoted=pd.crosstab(index=X['department'],columns=y)
department_vs_promoted.plot(kind='bar')

education_vs_promoted=pd.crosstab(index=X['education'],columns=y)
education_vs_promoted.plot(kind='bar')

gender_vs_promoted=pd.crosstab(index=X['gender'],columns=y)
gender_vs_promoted.plot(kind='bar')

age_vs_promoted=pd.crosstab(index=X['age'],columns=y)
age_vs_promoted.plot(kind='bar')

recruitment_channel_vs_promoted=pd.crosstab(index=X['recruitment_channel'],columns=y)
recruitment_channel_vs_promoted.plot(kind='bar',title='Not promoted eqully')

KPI_vs_promoted=pd.crosstab(index=X['KPIs_met >80%'],columns=y)
KPI_vs_promoted.plot(kind='bar')

award_vs_promoted=pd.crosstab(index=X['awards_won?'],columns=y)
award_vs_promoted.plot(kind='bar')

noTraining_vs_promoted=pd.crosstab(index=X['no_of_trainings'],columns=y)
noTraining_vs_promoted.plot(kind='bar')

dataset.boxplot(column='age',by='is_promoted')
dataset.boxplot(column='no_of_trainings',by='is_promoted')
dataset.boxplot(column='length_of_service',by='is_promoted')
