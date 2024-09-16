-- vehicle_speed_snapshot
-- SELECT
--   (cast(to_unixtime(timestamp_ms) * 1000 as bigint) / 100) * 100 timechunk,
--   avg("wheel_speed"."mean_km_per_hour") wsm_avg
-- FROM ree_cloud_data_{{ environment }}.telemetry
-- WHERE cast(to_unixtime(timestamp_ms) * 1000 as bigint) BETWEEN {{snapshot}} - (1000000 * 10) AND {{snapshot}} + (1000000 * 10)
-- AND session_id = '{{session_id}}'
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
   wheel_speed.mean_km_per_hour
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and wheel_speed is not null
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
