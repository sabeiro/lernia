-- rtp_stats_lost_late_snapshot
-- SELECT
--   (cast(to_unixtime(timestamp_ms) * 1000 as bigint) / 100) * 100 timechunk,
--   avg(rtp_stats.lost) lost,
--   avg(rtp_stats.late) late
-- FROM ree_cloud_data_{{ environment }}.telemetry
-- WHERE cast(to_unixtime(timestamp_ms) * 1000 as bigint) BETWEEN {{snapshot}} - (1000000 * 10) AND {{snapshot}} + (1000000 * 10)
-- -- WHERE e2e_latency.camera_index = 1
-- AND session_id = '{{ session_id }}'
-- GROUP BY 1 ORDER BY 1 ASC LIMIT 10000
with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
)
select
   to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
   rtp_stats.lost lost,
   rtp_stats.late late,
   rtp_stats.camera_index
from ree_cloud_data_{{ environment }}.telemetry TABLESAMPLE BERNOULLI (20)
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and rtp_stats is not null
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
