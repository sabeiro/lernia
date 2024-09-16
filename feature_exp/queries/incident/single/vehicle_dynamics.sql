with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
), timebuckets as (
select
    to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
    floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
    steering_wheel_state.brake / -655.35 as td_brake,
    steering_wheel_state.throttle / 655.35 as td_throttle,
    vehicle_physics.longitudinal_force_m_per_sec_squared as lon_force_m_per_sec_squared
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and (steering_wheel_state is not null or vehicle_physics is not null)
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
)

select 
    avg(td_brake) as avg_td_brake,
    avg(td_throttle) as avg_td_throttle,
    avg(lon_force_m_per_sec_squared) as avg_lon_force_m_per_sec_squared,
    {{snapshot}},
    timebucket,
    timebucket*1000 - {{snapshot}} as ts
from timebuckets
group by timebucket
order by timebucket

