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
    latency_camera.latency_ms as camera_latency_ms,
    (joystick_latency.joystick_latency_ns/1000000) as command_latency_ms,
    e2e_latency.latency_ms as e2e_latency_ms
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and (latency_camera is not null or joystick_latency is not null or e2e_latency is not null)
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
)
select 
    max(camera_latency_ms) as max_camera_latency_ms,
    max(command_latency_ms) as max_command_latency_ms,
    max(e2e_latency_ms) as max_e2e_latency_ms,
    {{snapshot}},
    timebucket,
    timebucket*1000 - {{snapshot}} as ts
    -- from_unixtime(timebucket)  as ts
    -- date_add('millisecond', (timebucket%100)*10 ,from_unixtime(timebucket/100))  as ts
from timebuckets
group by timebucket
order by timebucket

