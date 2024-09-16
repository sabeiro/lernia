with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
)
select
    to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
    steering_interval_max.interval_ms as interval_ms
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and steering_interval_max is not null
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms

