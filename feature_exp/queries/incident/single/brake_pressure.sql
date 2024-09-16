-- brake_snapshot
-- SELECT
--   (cast(to_unixtime(timestamp_ms) * 1000 as bigint) / 100) * 100 timechunk,
--   avg(steering_wheel_state.brake) avg_steering_wheel_state_brake,
--   avg(brake_pressure.brake_pressure) avg_brake_pressure
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
), value_partitions as (
select
    timestamp_ms,
    steering_wheel_state,
    sum(case when steering_wheel_state is null then 0 else 1 end) over (order by timestamp_ms) as steering_wheel_state_partition,
    brake_pressure,
    sum(case when brake_pressure is null then 0 else 1 end) over (order by timestamp_ms) as brake_pressure_partition
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and (steering_wheel_state is not null or brake_pressure is not null)
    and abs({{snapshot}}/1000 - to_unixtime(timestamp_ms)) < 10
order by timestamp_ms
), rolling_values as (
select
    timestamp_ms,
    first_value(steering_wheel_state) over (partition by steering_wheel_state_partition order by timestamp_ms) as steering_wheel_state,
    first_value(brake_pressure) over (partition by brake_pressure_partition order by timestamp_ms) as brake_pressure
from value_partitions
order by timestamp_ms
)
select
    to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
    steering_wheel_state.brake / 655.35 as control_brake,
    brake_pressure.brake_pressure as vehicle_brake
from rolling_values
order by timestamp_ms

