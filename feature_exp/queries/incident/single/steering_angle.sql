-- steering_snapshot
-- SELECT
--   (cast(to_unixtime(timestamp_ms) * 1000 as bigint) / 100) * 100 timechunk,
--   avg(steering_wheel_state.steering_deg) avg_steering_wheel_state,
--   avg(steering_angle.angle_deg) avg_steering_angle
-- FROM ree_cloud_data_{{ environment }}.telemetry
-- WHERE cast(to_unixtime(timestamp_ms) * 1000 as bigint) BETWEEN {{snapshot}} - (1000000 * 10) AND {{snapshot}} + (1000000 * 10)
-- AND session_id = '{{ session_id }}'
-- GROUP BY 1 ORDER BY 1 ASC LIMIT 10000

with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
), timebuckets as (
select
    timestamp_ms,
    floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
    steering_wheel_state.steering_deg steering_wheel_deg,
    steering_angle.angle_deg steering_angle_deg
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and (steering_wheel_state is not null or steering_angle is not null)
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
)
select 
    avg(steering_wheel_deg)/32767.0 * 420.0 as steering_wheel_deg,
    avg(steering_angle_deg) as steering_angle_deg,
    timebucket*1000 - {{snapshot}} as ts
from timebuckets
group by timebucket
order by timebucket

