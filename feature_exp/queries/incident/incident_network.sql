with session_info as (
    select 
        vehicle_id
    from
        ree_cloud_data_{environment}.telemetry
    where
        session_id = '{session_id}'
        and dt = date_format(from_unixtime({snapshot}/1000), '%Y-%m-%d')
    limit 1    
)
select 
    to_unixtime(timestamp_ms)*1000 - {snapshot} as ts,
    name,
    cast(bytes*8 as double)/interval_duration as mbit_second,
    case when packets = 0 then null else max_inter_packet_arrival_time/1000 end as max_inter_packet_arrival_time
from
    ree_cloud_data_prod.network_log 
where
    dt = date_format(from_unixtime({snapshot}/1000), '%Y-%m-%d')
    and  vehicle_id in (select vehicle_id from session_info)
    and timestamp_ms between from_unixtime({snapshot} /1000) - interval '{time_horizon_sec}' second and from_unixtime({snapshot}/1000) + interval '{time_horizon_sec}' second

order by timestamp_ms

    

