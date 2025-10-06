import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

ads_clients_data = pd.read_csv(r"C:\Users\Пользователь\Desktop\6_ads\ads_clients_data.csv", encoding='windows-1251')
ads_data         = pd.read_csv(r"C:\Users\Пользователь\Desktop\6_ads\ads_data.csv", encoding='windows-1251')

#print(ads_data['event'].unique())  'view' 'click'

ads_clients_data["create_date"] = pd.to_datetime(ads_clients_data["create_date"])
ads_data["date"]                = pd.to_datetime(ads_data["date"])
events_count                    = ads_data.groupby(['ad_id', 'event']).size().unstack(fill_value=0)

avg_click = round(events_count['click'].mean())
avg_views = round(events_count['view'].mean())
print('Среднее количество кликов за все время:', avg_click)
print('Среднее количество просмотров за все время:', avg_views)

# график распределения показов на объявление за весь период
sns.histplot(events_count['view'],
             bins=30,
             kde=True)
plt.title("Распределение количества показов на объявление за весь период")
plt.xlabel("Количество показов на объявление")
plt.ylabel("Количество объявлений")
#plt.show()

# boxplot (чтобы увидеть выбросы)
sns.boxplot(x=events_count['view'])
plt.title("Boxplot количества показов на объявление")
plt.xlabel("Количество показов")
#plt.show()

# фильтруем только показы

views_per_day = (
    ads_data.query("event == 'view'")
    .groupby("date")
    .size()
    .reset_index(name="views")
)

# Считаем скользящее среднее с окном 2
views_per_day["rolling_mean_2"] = (views_per_day["views"]
                                   .rolling(window=2)
                                   .mean())

# Ищем значение за 6 апреля 2019 года
value_apr6 = round(
    views_per_day
    .loc[views_per_day["date"] == "2019-04-06", "rolling_mean_2"]
    .values[0]
)

print("Скользящее среднее показов за 6 апреля 2019 года:", value_apr6)

# считаем разницу между скользящим и обычным средними
views_per_day["abs_diff"] = (views_per_day["rolling_mean_2"] - avg_views).abs()

# Находим день с наибольшей разницей
max_diff_day   = views_per_day.loc[views_per_day['abs_diff'].idxmax(), 'date']
max_diff_value = views_per_day["abs_diff"].max()
print(f"День с наибольшей разницей: {max_diff_day.date()} (разница = {max_diff_value:.0f})")


plt.figure(figsize=(12,6))
plt.plot(views_per_day["date"],
         views_per_day["views"],
         label="Показы в день",
         marker='o')
plt.plot(views_per_day["date"],
         views_per_day["rolling_mean_2"],
         label="Скользящее среднее (окно=2)",
         marker='o')
plt.axhline(avg_views,
            color='r',
            linestyle='--',
            label="Арифметическое среднее")
plt.legend()
plt.title("Показы и скользящее среднее по дням\n")
plt.xlabel("Дата")
plt.ylabel("Количество показов\n")
plt.grid(True)
#plt.show()

# Фильтруем показы только за найденный аномальный день
views_on_max_diff_day = (
    ads_data.query("event == 'view' and date == @max_diff_day")
    .groupby("ad_id")
    .size()
    .reset_index(name="views")
)

# Находим объявление с макс. и мин. показами в этот день
ad_with_max_views = views_on_max_diff_day.loc[views_on_max_diff_day['views'].idxmax()]
ad_with_min_views = views_on_max_diff_day.loc[views_on_max_diff_day['views'].idxmin()]

print("\n")
print(f"Объявление с МАКС. количеством показов ({max_diff_day.date()}): ad_id={ad_with_max_views['ad_id']}, "
      f"показы={ad_with_max_views['views']}")
print(f"Объявление с МИН. количеством показов ({max_diff_day.date()}): ad_id={ad_with_min_views['ad_id']}, "
      f"показы={ad_with_min_views['views']}")

sns.histplot(views_on_max_diff_day['views'], bins=20, kde=True)
plt.title(f"Распределение показов по объявлениям за {max_diff_day.date()}")
plt.xlabel("Количество показов")
plt.ylabel("Количество объявлений")
plt.show()

# Фильтруем только показы (view) и ищем первый показ по каждому клиенту
first_ad_dates = (ads_data.query("event == 'view'")
                  .groupby('client_union_id')['date']
                  .min()
                  .reset_index(name='first_ad_date'))

# Соединяем с данными о клиентах
clients_with_first_ad = ads_clients_data.merge(first_ad_dates, on="client_union_id", how="inner")

# Считаем разницу в днях между датой создания клиента и первым показом
clients_with_first_ad["days_to_first_ad"] = (
    clients_with_first_ad["first_ad_date"] - clients_with_first_ad["create_date"]
).dt.days

# Среднее количество дней
avg_days = clients_with_first_ad["days_to_first_ad"].mean()

print(f"Среднее количество дней от создания клиента до первого показа: {avg_days:.2f}")

# Фильтруем только тех, у кого реклама запущена ≤ 365 дней
conversion_count = clients_with_first_ad[
    (clients_with_first_ad["days_to_first_ad"].notna()) &
    (clients_with_first_ad["days_to_first_ad"] <= 365)].shape[0]

# Общее количество клиентов
total_clients = ads_clients_data.shape[0]

# Конверсия
conversion_rate = (conversion_count / total_clients) * 100

print(f"Конверсия клиентов в запуск первой рекламы ≤ 365 дней: {conversion_rate:.2f}%")
print(f"(Из {total_clients} клиентов — {conversion_count} запустили рекламу ≤ 365 дней)")

# Определяем интервалы
bins = [0, 30, 90, 180, 365]

# Классифицируем клиентов по "дням до первого запуска"
clients_with_first_ad["days_bin"] = pd.cut(
    clients_with_first_ad["days_to_first_ad"],
    bins=bins,
    right=True,   # включаем правую границу (0-30, 30-90 и т.д.)
    include_lowest=True
)

# Считаем количество уникальных клиентов в каждом интервале
clients_per_bin = clients_with_first_ad.groupby("days_bin")["client_union_id"].nunique().reset_index(name="unique_clients")

print(clients_per_bin)

# Сколько клиентов запустили рекламу в первый месяц (0–30 дней)
interval_dict = dict(zip(clients_per_bin["days_bin"], clients_per_bin["unique_clients"]))
first_month_clients = interval_dict.get(pd.Interval(0, 30, closed="right"), 0)

print(f"Количество клиентов, запустивших рекламу в первые 30 дней: {first_month_clients}")

# Преобразуем интервалы в строки
clients_per_bin["days_bin_str"] = clients_per_bin["days_bin"].astype(str)



fig = px.bar(
    clients_per_bin,
    x="days_bin_str",   # теперь строки вместо Interval
    y="unique_clients",
    title="Количество уникальных клиентов по времени запуска первой рекламы",
    labels={"days_bin_str": "Интервал (дни)", "unique_clients": "Уникальные клиенты"},
    text="unique_clients"
)
fig.update_traces(textposition="outside")
fig.show()













