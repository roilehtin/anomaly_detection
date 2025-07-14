import logging
import pandas as pd
from typing import Dict, Tuple
from geopy.distance import geodesic

logging.basicConfig(
    level=logging.WARNING,  # Минимум логов: предупреждения и ошибки
    format='%(asctime)s [%(levelname)s] %(message)s'
)

def get_russian_holiday(ts: pd.Timestamp) -> str:
    date_val = ts.date()
    if date_val.month == 1 and date_val.day in range(1, 9):
        return 'new_year'
    if date_val.month == 4 and date_val.day in range(12, 20):
        return 'easter'
    if date_val.month == 6 and date_val.day == 12:
        return 'russia_day'
    if date_val.month == 7 and date_val.day == 1:
        return 'study_day'
    return 'none'

def get_season(ts: pd.Timestamp) -> str:
    month = ts.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'autumn'

def build_segment_percentiles_with_season(df_hist: pd.DataFrame, metrics: list, percentile: float = 0.99) -> pd.DataFrame:
    df = df_hist.copy()
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['holiday'] = df['datetime'].apply(get_russian_holiday)
    df['season'] = df['datetime'].apply(get_season)

    group_cols = ['hour', 'weekday', 'holiday', 'season']
    stats = (
        df.groupby(group_cols)[metrics]
        .quantile(percentile)
        .reset_index()
    )
    stats.columns = group_cols + [f"{m}_p{int(percentile*100)}" for m in metrics]
    return stats

def detect_anomalies_by_time_segment(
        df_hist: pd.DataFrame,
        df_current: pd.DataFrame,
        metrics: list,
        percentile: float = 0.99,
        min_metrics: int = 2
    ) -> pd.DataFrame:
    stats = build_segment_percentiles_with_season(df_hist, metrics, percentile)
    df = df_current.copy()
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday
    df['holiday'] = df['datetime'].apply(get_russian_holiday)
    df['season'] = df['datetime'].apply(get_season)

    def check(row):
        segment = stats[
            (stats['hour'] == row['hour']) &
            (stats['weekday'] == row['weekday']) &
            (stats['holiday'] == row['holiday']) &
            (stats['season'] == row['season'])
        ]
        if segment.empty:
            return False
        anomaly_count = 0
        for metric in metrics:
            perc_col = f"{metric}_p{int(percentile*100)}"
            perc_val = segment[perc_col].values[0]
            if row[metric] > perc_val:
                anomaly_count += 1
        return anomaly_count >= min_metrics

    df['anomaly'] = df.apply(check, axis=1)
    return df

def filter_spatial_anomalies(
        df: pd.DataFrame,
        rsu_locations: Dict[int, Tuple[float, float]],
        radius_meters: float = 400,
        count_neighbours: int = 2
    ) -> pd.DataFrame:
    df = df.copy()
    df['anomaly_filtered'] = df['anomaly']

    grouped = df[df['anomaly']].groupby('datetime')

    for dt, group in grouped:
        rsu_ids = group['rsu_id'].unique()
        for rsu_id in rsu_ids:
            if rsu_id not in rsu_locations:
                continue
            coord_main = rsu_locations[rsu_id]
            neighbors = 0
            for other_id in rsu_ids:
                if other_id == rsu_id or other_id not in rsu_locations:
                    continue
                coord_other = rsu_locations[other_id]
                distance = geodesic(coord_main, coord_other).meters
                if distance <= radius_meters:
                    neighbors += 1
                if neighbors >= count_neighbours:
                    idxs = df[(df['datetime'] == dt) & (df['rsu_id'] == rsu_id)].index
                    df.loc[idxs, 'anomaly_filtered'] = False
                    break

    return df

def detect_anomalies_for_latest_timestamp(
        df_hist: pd.DataFrame,
        df_full: pd.DataFrame,
        metrics: list,
        threshold: float = 99,
        min_metrics: int = 2,
        radius_meters: float = 400,
        count_neighbours: int = 2
    ) -> pd.DataFrame:
    max_datetime = df_full['datetime'].max()
    df_latest = df_full[df_full['datetime'] == max_datetime].copy()

    df_latest = detect_anomalies_by_time_segment(
        df_hist,
        df_latest,
        metrics=metrics,
        percentile=threshold / 100,
        min_metrics=min_metrics
    )

    coords_df = df_full[['rsu_id', 'latitude', 'longitude']].drop_duplicates()
    rsu_locations = {row.rsu_id: (row.latitude, row.longitude) for row in coords_df.itertuples()}

    df_filtered = filter_spatial_anomalies(df_latest, rsu_locations, radius_meters, count_neighbours)

    return df_filtered[['rsu_id', 'datetime', 'anomaly_filtered']]
