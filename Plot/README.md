<img src=".\plot_images\plot_1.png" alt="plot_1" style="zoom: 67%;" />

```python
import matplotlib.pyplot as plt
plt.style.use('default')
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# mpl.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False  # 显示负号


fig = plt.figure(figsize=(20, 15), dpi=300)
ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((4, 2), (2, 0))
ax3 = plt.subplot2grid((4, 2), (2, 1))
ax4 = plt.subplot2grid((4, 2), (3, 0))
ax5 = plt.subplot2grid((4, 2), (3, 1))

# ax1
ax1.plot(df['Date'], df['Number of reported results'], 'k.', markersize=15, alpha=.3, label='$\mathrm{Number\ of\ reported\ results}$')
ax1.plot(fore_1['ds'], fore_1['yhat'],
         c='#00528c', linewidth=6, label=r'$y_1(t) = g_1(t) + s_1(t) + h_1(t) + \epsilon_{1t}$')

ax1.plot(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['yhat'],
         c='#5083C2', linewidth=6, label=r'$y_2(t) = g_2(t) + s_2(t) + h_2(t) + \epsilon_{1t}$')
ax1.plot(fore_2[fore_2['ds'] > '2022-12-31']['ds'], fore_2[fore_2['ds'] > '2022-12-31']['yhat'],
         c='#f15138', linewidth=6, label='$\mathrm{Forecast\ value}\ \hat y(t)$')

ax1.plot(fore_2['ds'], fore_2['floor'], 'k--', linewidth=1.5, alpha=0.7)


ax1.fill_between(fore_1['ds'], fore_1['yhat_lower'], fore_1['yhat_upper'], color='#cbe2f5')
ax1.fill_between(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['yhat_lower'], fore_2[fore_2['ds'] <= '2022-12-31']['yhat_upper'], color='#cbe2f5')
ax1.fill_between(fore_2[fore_2['ds'] > '2022-12-31']['ds'], fore_2[fore_2['ds'] > '2022-12-31']['yhat_lower'], fore_2[fore_2['ds'] > '2022-12-31']['yhat_upper'], color='#fdd2c1')

ax1.set_xlabel('(a) Predictions from our time series forecasting model', fontsize=22)
ax1.set_ylabel('Number of reported results', fontsize=30, weight='bold')
ax1.tick_params(axis='both', which='major', labelsize=18)

plt.rcParams['text.usetex'] = True
ax1.legend(loc='upper right', prop=font_legend_1)
plt.rcParams['text.usetex'] = False
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.5)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)


# 嵌入绘制局部放大图的坐标系
# axins_1
axins_1 = ax1.inset_axes((0.22, 0.5, 0.22, 0.4))

axins_1.plot(df['Date'], df['Number of reported results'], 'k.',
             markersize=6, alpha=.5, label='$\mathrm{Number\ of\ reported\ results}$')
axins_1.plot(fore_1['ds'], fore_1['yhat'], c='#00528c', linewidth=3)
axins_1.plot(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['yhat'],
             c='#5083C2', linewidth=3, label=r'$y_2(t) = g_2(t) + s_2(t) + h_2(t) + \epsilon_{1t}$')

axins_1.fill_between(fore_1['ds'], fore_1['yhat_lower'], fore_1['yhat_upper'], color='#cbe2f5')
axins_1.fill_between(fore_2[fore_2['ds'] <= '2022-12-31']['ds'],
                     fore_2[fore_2['ds'] <= '2022-12-31']['yhat_lower'],
                     fore_2[fore_2['ds'] <= '2022-12-31']['yhat_upper'], color='#cbe2f5')
axins_1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


for spine in axins_1.spines.values():
    spine.set_linewidth(3.5)

x_ratio = 0.3 # x轴显示范围的扩展比例
y_ratio = 0.4 # y轴显示范围的扩展比例

xlim0 = datetime(2022, 1, 20)
xlim1 = datetime(2022, 2, 16)

y_s = fore_1[fore_1['ds'] >= xlim0 ][fore_1['ds'] <= xlim1]['yhat']
ylim0 = min(y_s) - y_ratio * (max(y_s) - min(y_s))
ylim1 = max(y_s) + y_ratio * (max(y_s) - min(y_s))


# 调整子坐标系的显示范围
axins_1.set_xlim(xlim0, xlim1)
axins_1.set_ylim(ylim0, ylim1)

# 设置 x 轴的刻度为每 7 天一个刻度
locator = mdates.WeekdayLocator(byweekday=mdates.MO)  # 每周一作为刻度
formatter = mdates.DateFormatter('%m-%d')  # 设置日期格式

axins_1.xaxis.set_major_locator(locator)
axins_1.xaxis.set_major_formatter(formatter)
axins_1.tick_params(axis='both', which='major', labelsize=16)

mark_inset(ax1, axins_1, loc1=3, loc2=1, fc="none", ec='0.2', lw=2.5, zorder=10)


# axins_2
axins_2 = ax1.inset_axes((0.55, 0.25, 0.43, 0.4))
#
axins_2.plot(df['Date'], df['Number of reported results'], 'k.',
             markersize=6, alpha=.5, label='$\mathrm{Number\ of\ reported\ results}$')
axins_2.plot(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['yhat'],
             c='#5083C2', linewidth=3, label=r'$y_2(t) = g_2(t) + s_2(t) + h_2(t) + \epsilon_{1t}$')
axins_2.plot(fore_2[fore_2['ds'] > '2022-12-31']['ds'], fore_2[fore_2['ds'] > '2022-12-31']['yhat'],
             c='#f1553c', linewidth=3, label='$\mathrm{Forecast\ value}\ \hat y(t)$')


axins_2.fill_between(fore_2[fore_2['ds'] <= '2022-12-31']['ds'],
                     fore_2[fore_2['ds'] <= '2022-12-31']['yhat_lower'],
                     fore_2[fore_2['ds'] <= '2022-12-31']['yhat_upper'], color='#cbe2f5')
axins_2.fill_between(fore_2[fore_2['ds'] > '2022-12-31']['ds'],
                 fore_2[fore_2['ds'] > '2022-12-31']['yhat_lower'],
                 fore_2[fore_2['ds'] > '2022-12-31']['yhat_upper'], color='#fdd2c1')
axins_2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

for spine in axins_2.spines.values():
    spine.set_linewidth(3.5)

x_ratio = 0.3 # x轴显示范围的扩展比例
y_ratio = 0.3 # y轴显示范围的扩展比例
ratio_0 = 0.25

xlim0 = datetime(2022, 12, 1)
xlim1 = datetime(2023, 3, 12)

y_s = fore_2[fore_2['ds'] >= xlim0 ][fore_2['ds'] <= xlim1]['yhat']
ylim0 = min(y_s) - (y_ratio+ratio_0) * (max(y_s) - min(y_s))
ylim1 = max(y_s) + y_ratio * (max(y_s) - min(y_s))


# 调整子坐标系的显示范围
axins_2.set_xlim(xlim0, xlim1)
axins_2.set_ylim(ylim0, ylim1)

# 设置 x 轴的刻度为每 7 天一个刻度
locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=2)  # 每周一作为刻度
formatter = mdates.DateFormatter('%m-%d')  # 设置日期格式

axins_2.xaxis.set_major_locator(locator)
axins_2.xaxis.set_major_formatter(formatter)
axins_2.tick_params(axis='both', which='major', labelsize=16)

mark_inset(ax1, axins_2, loc1=3, loc2=4, fc="none", ec='0.2', lw=2.5, zorder=10)


# ax2
ax2.plot(fore_1['ds'], fore_1['trend'], c='#076fa2', label='$g_1(t)$', linewidth=2.5)
ax2.plot(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['trend'],
        c='#2ebfd2', linewidth=2.5, label='$g_2(t)$')
ax2.plot(fore_2[fore_2['ds'] > '2022-12-31']['ds'], fore_2[fore_2['ds'] > '2022-12-31']['trend'],
         c='#f15138', linewidth=2.5, label='$\mathrm{Forecast\ trend}\ \hat g(t)$')
y_min, y_max = ax2.get_ylim()

ax2.fill_between(fore_1['ds'], y_min, y_max, color='#cdebf5')
ax2.fill_between(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], y_min, y_max, color='#e3f4f4')
ax2.fill_between(fore_2[fore_2['ds'] > '2022-12-31']['ds'], y_min, y_max, color='#FFE5CA')

plt.rcParams['text.usetex'] = True
ax2.legend(loc='upper right', prop=font_legend_2)
plt.rcParams['text.usetex'] = False
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.5)
ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.set_xlabel('(b) Trend term function', fontsize=22)
ax2.set_ylabel('Trend of the reported results', fontsize=18, weight='bold')
ax2.tick_params(axis='both', which='major', labelsize=16)

ax2.spines['bottom'].set_linewidth(1.5)
ax2.spines['left'].set_linewidth(1.5)

# ax3
ax3.plot(fore_1['ds'], fore_1['holidays'], c='#076fa2', label='$h_1(t)$', linewidth=2.5)
ax3.plot(fore_2[fore_2['ds'] <= '2022-12-31']['ds'], fore_2[fore_2['ds'] <= '2022-12-31']['holidays'],
         c='#2ebfd2', linewidth=2.5, label='$h_2(t)$')
ax3.plot(fore_2[fore_2['ds'] > '2022-12-31']['ds'], fore_2[fore_2['ds'] > '2022-12-31']['holidays'],
         c='#f15138', linewidth=2.5, label=r'$\mathrm{Forecast\ external\ variable}\ \hat h(t)$')
y_min, y_max = ax2.get_ylim()

plt.rcParams['text.usetex'] = True
ax3.legend(loc='upper right', prop=font_legend_2)
plt.rcParams['text.usetex'] = False
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='y', alpha=0.5)
ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax3.set_xlabel('(c) External variable term', fontsize=22)
ax3.set_ylabel("External variables\' influence", fontsize=18, weight='bold')
ax3.tick_params(axis='both', which='major', labelsize=16)

ax3.spines['bottom'].set_linewidth(1.5)
ax3.spines['left'].set_linewidth(1.5)


# ax4
def plot_weekly(m, ax, uncertainty=True, weekly_start=0, start_date=None, name='weekly', color=None):
    days = (pd.date_range(start=start_date, periods=7) + pd.Timedelta(days=weekly_start))
    df_w = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_w)
    days = days.day_name()
    ax.plot(range(len(days)), seas[name], ls='-', c=color, linewidth=2.5)

    ax.grid(axis='y', alpha=0.5)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(days)

def seasonality_plot_df(m, ds):
    df_dict = {'ds': ds, 'cap': 1., 'floor': 0.}
    for name in m.extra_regressors:
        df_dict[name] = 0.
    # Activate all conditional seasonality columns
    for props in m.seasonalities.values():
        if props['condition_name'] is not None:
            df_dict[props['condition_name']] = True
    df = pd.DataFrame(df_dict)
    df = m.setup_dataframe(df)
    return df

plot_weekly(model_1, ax4, uncertainty=True, weekly_start=0, start_date='2017-01-01', color='#076fa2')

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(axis='y', alpha=0.5)
ax4.set_xlabel('(d) The periodic term for the previous period', fontsize=22)
ax4.set_ylabel('Weekly periodic of TIME 1', fontsize=18, weight='bold')
ax4.tick_params(axis='both', which='major', labelsize=16)
ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax4.spines['bottom'].set_linewidth(1.5)
ax4.spines['left'].set_linewidth(1.5)

# ax5
plot_weekly(model_2, ax5, uncertainty=True, weekly_start=0, start_date='2017-01-01', color='#2ebfd2')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.grid(axis='y', alpha=0.5)
ax5.set_xlabel('(e) The periodic term for the later period', fontsize=22)
ax5.set_ylabel('Weekly periodic of TIME 2', fontsize=18, weight='bold')
ax5.tick_params(axis='both', which='major', labelsize=16)

ax5.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

ax5.spines['bottom'].set_linewidth(1.5)
ax5.spines['left'].set_linewidth(1.5)

plt.tight_layout()
plt.show()
```



---



<img src="plot_images\plot_2.png" alt="plot_2" style="zoom: 50%;" />

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('default')
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
# mpl.rcParams['mathtext.default'] = 'regular'
plt.rcParams['axes.unicode_minus'] = False  # 显示负号

fig = plt.figure(figsize=(20, 8), dpi=100)
ax1 = plt.subplot2grid((2, 3), (0, 0), rowspan=1, colspan=3)
ax2 = plt.subplot2grid((2, 3), (1, 0))
ax3 = plt.subplot2grid((2, 3), (1, 1))
ax4 = plt.subplot2grid((2, 3), (1, 2))


# ax1.plot(df_1['Date'], df_1['Number of reported results'], 'k.')
# ax1.plot(df_t['ds'], df_t['yhat'], label=r'$\mathrm{Original\ model}$')
ax1.plot(df['Date'], df['Number of reported results'], 'k.', markersize=6, alpha=.3, label='Raw data')
ax1.plot(df_t['ds'], df_t['yhat'], label='Original model',
         color='#f1674b', linewidth=3)
ax1.plot(forecast_1['ds'], forecast_1['yhat'] + 7e3, label='Adding abnormal data',
         color='#f69779', linewidth=3)
ax1.plot(forecast_2['ds'], forecast_2['yhat'] + 5e3, label='Adding Gaussian noise',
         color='#2dc0d4', linewidth=3)
ax1.plot(forecast_3['ds'], forecast_3['yhat'] - 1e3, label='Missing data',
         color='#076fa2', linewidth=3)
ax1.plot(np.full(100, datetime(2023, 3, 1)), np.linspace(-5e4, 3.8e5, 100), 'r--', linewidth=3)


ax1.set(ylim=(-5e4, 3.8e5))
# ax1.set_xlabel('(a) Predictions from our time series forecasting model', fontsize=22)
ax1.set_ylabel('Number of reported results', fontsize=20, weight='bold')
ax1.tick_params(axis='both', which='major', labelsize=18)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.5)
ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

# plt.rcParams['text.usetex'] = True
ax1.legend(loc='upper right', prop=font_legend_1)
# plt.rcParams['text.usetex'] = False
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)


# zoom_effect01(ax2, ax1,
#               mdates.date2num(datetime(2022, 4, 5)), mdates.date2num(datetime(2022, 6, 5)),
#               0, 1)

ax2.plot(df_3['Date'], df_3['Number of reported results'], 'ko', markersize=7, alpha=.3, label='Raw data with missing data')
ax2.plot(forecast_3['ds'], forecast_3['yhat'], label='Prediction on missing data',
         color='#076fa2', linewidth=4)
ax2.fill_between(forecast_3['ds'], forecast_3['yhat_lower'], forecast_3['yhat_upper'], color='k', alpha=0.1)

ax2.set(xlim=(datetime(2022, 3, 20), datetime(2022, 6, 25)), ylim=(40000, 150000))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.set_xlabel('(a) Prediction on missing data', fontsize=22)
ax2.set_ylabel('Number of reported results', fontsize=18, weight='bold')
ax2.tick_params(axis='both', which='major', labelsize=14)

ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax2.legend(loc='upper right', prop=font_legend_2)




ax3.scatter(df_1['Date'], df_1['Number of reported results'], color='#7fc7d9', s=30, alpha=.7, label='Anomalous data')
ax3.scatter(df_2['Date'], df_2['Number of reported results'], color='#f69a81', s=30, alpha=.7, label='Noisy data')
ax3.plot(forecast_1['ds'], forecast_1['yhat'] + 9e3,
         label='Prediction - Anomalous data', color='#365486', linewidth=3)
ax3.plot(forecast_2['ds'], forecast_2['yhat'] + 5e3,
         label='Prediction - Noisy data', color='#f1553c', linewidth=3)
ax3.fill_between(forecast_1['ds'], forecast_1['yhat_lower'], forecast_1['yhat_upper'], color='k', alpha=0.05)
ax3.fill_between(forecast_2['ds'], forecast_2['yhat_lower'], forecast_2['yhat_upper'], color='k', alpha=0.05)


ax3.set(xlim=(datetime(2022, 8, 15), datetime(2022, 12, 31)), ylim=(-1e4, 100000))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax3.set_xlabel('(b) Prediction on outliers and noisy data', fontsize=22)
# ax3.set_ylabel('Number of reported results', fontsize=18, weight='bold')
ax3.tick_params(axis='both', which='major', labelsize=14)

ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax3.legend(loc='upper right', prop=font_legend_2)



ax4.plot(df_t['ds'], df_t['yhat'], label='Original model',
         color='#bd2335', linewidth=3)
ax4.plot(forecast_1['ds'], forecast_1['yhat'] * 1.3, label='Adding abnormal data',
         color='#f7b59b', linewidth=3)
ax4.plot(forecast_2['ds'], forecast_2['yhat'] * 1.2, label='Adding Gaussian noise',
         color='#b6dce6', linewidth=3)
ax4.plot(forecast_3['ds'], forecast_3['yhat'] * 0.95, label='Missing data',
         color='#347cb3', linewidth=3)
ax4.plot(np.full(100, datetime(2023, 3, 1)), np.linspace(0, 1e5, 100), 'r--', linewidth=5)

x_ratio = 0.3 # x轴显示范围的扩展比例
y_ratio = 0.3 # y轴显示范围的扩展比例
ratio_0 = 0.15

xlim0 = datetime(2023, 2, 17)
xlim1 = datetime(2023, 3, 10)

y_s = df_t[df_t['ds'] >= xlim0 ][df_t['ds'] <= xlim1]['yhat']
ylim0 = min(y_s) - (y_ratio+ratio_0) * (max(y_s) - min(y_s))
ylim1 = max(y_s) + y_ratio * (max(y_s) - min(y_s))

# 调整子坐标系的显示范围
ax4.set_xlim(xlim0, xlim1)
ax4.set_ylim(ylim0, ylim1)
# ax4.set(xlim=(datetime(2023, 2, 15), datetime(2023, 3, 15)), ylim=(-1e4, 100000))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.set_xlabel('(c) Projections of the future', fontsize=22)
ax4.tick_params(axis='both', which='major', labelsize=14)

ax4.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax4.legend(loc='upper right', prop=font_legend_2)


y_min, y_max = ax1.get_ylim()
ax1.fill_between(df_t[(df_t['ds'] >= '2022-03-20') & (df_t['ds'] <= '2022-06-25')]['ds'], y_min, y_max, color='#d2e3f0')
ax1.fill_between(df_t[(df_t['ds'] >= '2022-08-15') & (df_t['ds'] <= '2022-12-31')]['ds'], y_min, y_max, color='#d2e3f0')
ax1.fill_between(df_t[(df_t['ds'] >= '2023-02-17') & (df_t['ds'] <= '2023-03-10')]['ds'], y_min, y_max, color='#d2e3f0')

plt.rcParams['text.usetex'] = True
ax1.text(datetime(2022, 5, 7), 3e5, r'$\textbf{Make some data missing}$', fontsize=20,  ha='center')
ax1.text(datetime(2022, 10, 20), 3e5, r'$\textbf{Add abnormal data}$', fontsize=20,  ha='center')
ax1.text(datetime(2022, 10, 20), 2.4e5, r'$\textbf{Add Gaussian noise} \sim \mathcal N(0, \sigma^2),$', fontsize=20,  ha='center')
ax1.text(datetime(2022, 11, 13), 1.9e5, r'$\sigma^2 \in [8000, 30000].$', fontsize=20,  ha='center')
plt.rcParams['text.usetex'] = False

plt.tight_layout()
plt.show()
```



---



<img src="plot_images\plot_3.png" alt="plot_3" style="zoom: 60%;" />

```python
plt.figure(figsize=(14, 5), dpi=80)
plt.plot(range(1, 11), bics, "o-", c='#41a0dd',
         label="BIC", zorder=10, linewidth=4, markersize=10)
plt.plot(range(1, 11), aics, "o-", c='#d22d3f',
         label="AIC", zorder=10, linewidth=4, markersize=10)
plt.fill_between(range(1, 11), -27000, bics, color='#ecf5fc')
plt.fill_between(range(1, 11), -27000, aics, color='#faeaeb')
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.grid(alpha=0.6)
plt.legend(fontsize=25)
plt.grid(alpha=0.6)
plt.xticks(fontsize=23)
plt.ylabel('BIC and AIC for different $K$', fontsize=25, weight='bold')
plt.xlabel('The number of clusters $K$', fontsize=25, weight='bold')
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```



---



<img src="plot_images\plot_4.png" alt="plot_4" style="zoom: 33%;" />

```python
# 计算每个网格点的密度值
densities = np.exp(gmm_2.score_samples(grid))
densities = densities.reshape(x.shape)

# 绘制3D密度图
fig = plt.figure(figsize=(10, 7), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, densities, cmap=plt.get_cmap('Blues'), antialiased=False, linewidth=0, alpha=0.75)

ax.contourf(x, y, densities, linewidths=0.2, offset=7.5, cmap=plt.get_cmap('Blues'), alpha=1)
ax.contour(x, y, densities, linewidths=0.1, offset=7.5, cmap=plt.get_cmap('Blues'))

ax.scatter(X_train_2[y_pred_02[0]==0]['Proportion of hard'], X_train_2[y_pred_02[0]==0]['score'], -1,
           c='#08a3b9', s=10, zorder=10)
ax.scatter(X_train_2[y_pred_02[0]==1]['Proportion of hard'], X_train_2[y_pred_02[0]==1]['score'], -1,
           c='#28437a', s=10, zorder=10)
ax.scatter(X_train_2[y_pred_02[0]==2]['Proportion of hard'], X_train_2[y_pred_02[0]==2]['score'], -1,
           c='#d76735', s=10, zorder=10)
ax.plot(X_test['Proportion of hard'], X_test['score'], -0.9, 'r*', markersize=25, zorder=18)
ax.plot(X_test['Proportion of hard'], X_test['score'], 7.51, 'ro', markersize=8, zorder=20)
ax.plot(np.full(100, X_test['Proportion of hard'].ravel()),
        np.full(100, X_test['score'].ravel()),
        np.linspace(-0.9, 7.51, 100), '--', zorder=19)
ax.text(2.6, 0.15, 7, 'EERIE', fontsize=16, color='k', zorder=21)
ax.set_xlabel('Proportion of hard mode', fontsize=14, weight='bold', rotation=0)
ax.set_ylabel('Weighted number of attempts', fontsize=14, weight='bold', rotation=0)
ax.set_zlabel('Probability density', fontsize=14, weight='bold', rotation=0)
ax.view_init(elev=20, azim=110)
ax.set_zlim(bottom=-1.2, top=8)

plt.tight_layout()
plt.show()
```



---



<img src="plot_images\plot_5.png" alt="plot_5" style="zoom:33%;" />

```python
xx = np.linspace(-13, 9, 20)
yy = np.linspace(-1, 5, 20)
xx, yy = np.meshgrid(xx, yy)
grid = np.c_[xx.ravel(), yy.ravel()]
mu, cov = gaussian_process.predict(grid, return_std=True)
z = mu.reshape(xx.shape)

fig = plt.figure(figsize=(7, 5), dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, z, cmap=cm.coolwarm, linewidth=0, alpha=.8, antialiased=False)
colors = np.array([['#a7b4ec' for _ in range(20)] for _ in range(20)])
ax.plot_surface(xx, yy, np.full(400, 0).reshape(xx.shape), facecolors=colors, alpha=0.2, linewidth=0, antialiased=True)
ax.scatter(np.asarray(X_train_all_2)[:, 0], np.asarray(X_train_all_2)[:, 1], y_train_all_2, c=y_train_all_2, cmap=cm.coolwarm)
ax.contourf(xx, yy, z, zdir='z', offset=-3.5, cmap=cm.coolwarm, alpha=.5)

ax.view_init(elev=20, azim=120)
# ax.set_xlim(-40, 40)
ax.set_zlim(bottom=-3.5, top=3)
plt.rcParams['text.usetex'] = True
ax.set_xlabel('$W_3$', fontsize=14, weight='bold', rotation=0)
ax.set_ylabel('$H_2^{(6)}$', fontsize=14, weight='bold', rotation=0)
ax.set_zlabel('Momentum difference', fontsize=14, weight='bold', rotation=0)
plt.rcParams['text.usetex'] = False
ax.grid(False)
# ax.set_facecolor(None)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

plt.tight_layout()
plt.show()
```

