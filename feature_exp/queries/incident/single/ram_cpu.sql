with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
), timebuckets as (
select
    floor(to_unixtime(timestamp_ms)*100)/100 as timebucket,
    *
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and (control_room_ram_usage is not null or vehicle_ram_usage is not null or control_room_cpu_usage is not null or vehicle_cpu_usage is not null)
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
)
select 
  max(control_room_ram_usage.ram_usage_percent) control_room_ram,
  max(control_room_cpu_usage.cpu_usage_percent) control_room_cpu,
  max(vehicle_ram_usage.ram_usage_percent) vehicle_ram,
  max(vehicle_cpu_usage.cpu_usage_percent) vehicle_cpu,
  timebucket*1000 - {{snapshot}} as ts
from timebuckets
group by timebucket
order by timebucket

