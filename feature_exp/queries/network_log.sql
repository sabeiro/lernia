with ts_range as (
select distinct ts
from ree_cloud_data_{environment}.network_log
where 
    dt = date_format(date_trunc('day', from_unixtime({snapshot})), '%Y-%m-%d')
    and vehicle_id = '{vehicle_id}'
), timebuckets as (
select
	timestamp_ms,
	floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
	
	name,
	
	packets,
	bytes, 
	max_ttl,
	median_inter_packet_arrival_time as arrival_time,
	interval_duration

from ree_cloud_data_{environment}.network_log
where
    ts in (select ts from ts_range)
    and vehicle_id = '{vehicle_id}'
    -- and (latency_camera is not null or joystick_latency is not null or e2e_latency is not null)
    and abs({snapshot} - to_unixtime(timestamp_ms)) < {time_horizon_sec}
order by timestamp_ms
)
select 
	timebucket,
	-- timebucket - {snapshot} as ts,
	name, 

	-- avg(packets) as packets,
	avg(bytes) as bytes, 
	-- avg(max_ttl) as ttl,
	avg(arrival_time) as arrival_time
	-- avg(interval_duration) as interval_duration

    -- from_unixtime(timebucket)  as ts
    -- date_add('millisecond', (timebucket%100)*10 ,from_unixtime(timebucket/100))  as ts
from timebuckets
group by timebucket, name
order by timebucket
