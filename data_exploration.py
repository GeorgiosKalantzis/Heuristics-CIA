
"""
Data exploration
"""
import seaborn as sns
import matplotlib.pyplot as plt


# HEATMAP

cols = ['deviceWidth', 'deviceHeight', 'direction',
       'horizontalAcceleration', 'horizontalMeanPosition',
       'horizontalTraceLength', 'traceCoefDetermination',
       'traceMeanAbsoluteError', 'traceMeanSquaredError',
       'traceMedianAbsoluteError', 'traceSlope', 'verticalAcceleration',
       'verticalMeanPosition', 'verticalTraceLength', 'start_x', 'start_y',
       'stop_x', 'stop_y', 'median_vel_3fpts', 'median_vel_3lpts',
       'mid_stroke_area', 'angular_dispersion']

pp = sns.pairplot(data[cols], size=1.8, aspect=1.8,
                  plot_kws=dict(edgecolor="k", linewidth=0.5),
                  diag_kind="kde", diag_kws=dict(shade=True))

fig = pp.fig 
fig.subplots_adjust(top=0.93, wspace=0.3)


# CORRELATION

f, ax = plt.subplots(figsize=(10, 6))
corr = data.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle(' Heatmap', fontsize=14)



# ENTROPY

# Entropy with respect to tagrget
importances = mutual_info_classif(x,y)

feat_importances = pd.Series(importances, x.columns[0:len(x.columns)])
feat_importances.plot(kind='barh', color='blue') #plot info gain of each features
plt.show()

# select the number of features you want to retain.
select_k = 20 #whatever we want

# create the SelectKBest with the mutual info(info gain) strategy.
selection = SelectKBest(mutual_info_classif, k=select_k).fit(x, y)

#plot the scores
plt.bar([i for i in range(len(selection.scores_))], selection.scores_)
plt.show()

# display the retained features.
features = x.columns[selection.get_support()]
print(features)
