with ts_range as (
select distinct ts
from ree_cloud_data_{environment}.telemetry
where 
	ts='{ts}'
), timebuckets as (
select
	timestamp_ms,
	floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
	
	session_id,
	vehicle_id,

	longitude,
	latitude,

	radar_track.object_distance_m as object_distance_m,
	brake_pressure,
	vehicle_physics.lateral_force_m_per_sec_squared as lat_force,
	vehicle_physics.longitudinal_force_m_per_sec_squared as lon_force,
	vehicle_physics.yaw_rate_deg_per_sec as yaw_rate_deg,
	steering_wheel_state.steering_deg steering_wheel_deg,
    	steering_angle.angle_deg steering_angle_deg,
	wheel_speed.mean_km_per_hour as wheel_speed,
	vehicle_ping.ping_ms as vehicle_ping_ms,
	rtp_stats.lost as rtp_lost,
   	rtp_stats.late as rtp_late,
	modem_report.modem_report_data[1].tunnel_delta_rx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[1].tunnel_delta_interval_ns / 1000000000.0)) AS modem0_rx,
	modem_report.modem_report_data[2].tunnel_delta_rx / (modem_report.modem_report_data[2].tunnel_delta_interval_sec + (modem_report.modem_report_data[2].tunnel_delta_interval_ns / 1000000000.0)) AS modem1_rx,
	modem_report.modem_report_data[3].tunnel_delta_rx / (modem_report.modem_report_data[3].tunnel_delta_interval_sec + (modem_report.modem_report_data[3].tunnel_delta_interval_ns / 1000000000.0)) AS modem2_rx,
	modem_report.modem_report_data[4].tunnel_delta_rx / (modem_report.modem_report_data[4].tunnel_delta_interval_sec + (modem_report.modem_report_data[4].tunnel_delta_interval_ns / 1000000000.0)) AS modem3_rx,
	modem_report.modem_report_data[1].tunnel_delta_tx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[1].tunnel_delta_interval_ns / 1000000000.0)) AS modem0_tx,
	modem_report.modem_report_data[2].tunnel_delta_tx / (modem_report.modem_report_data[2].tunnel_delta_interval_sec + (modem_report.modem_report_data[2].tunnel_delta_interval_ns / 1000000000.0)) AS modem1_tx,
	modem_report.modem_report_data[3].tunnel_delta_tx / (modem_report.modem_report_data[3].tunnel_delta_interval_sec + (modem_report.modem_report_data[3].tunnel_delta_interval_ns / 1000000000.0)) AS modem2_tx,
	modem_report.modem_report_data[4].tunnel_delta_tx / (modem_report.modem_report_data[4].tunnel_delta_interval_sec + (modem_report.modem_report_data[4].tunnel_delta_interval_ns / 1000000000.0)) AS modem3_tx,
	modem_report.modem_report_data[1].tunnel_rtt AS modem0_rtt,
	modem_report.modem_report_data[2].tunnel_rtt AS modem1_rtt,
	modem_report.modem_report_data[3].tunnel_rtt AS modem2_rtt,
	modem_report.modem_report_data[4].tunnel_rtt AS modem3_rtt,
	jitter.jitter_ms as camera_jitter_ms,
	control_room_ram_usage.ram_usage_percent as control_room_ram,
	control_room_cpu_usage.cpu_usage_percent as control_room_cpu,
  	vehicle_ram_usage.ram_usage_percent as vehicle_ram,
  	vehicle_cpu_usage.cpu_usage_percent as vehicle_cpu,

	latency_camera.latency_ms as camera_latency_ms,
	latency_camera.camera_index as camera_index,	
    	joystick_latency.joystick_latency_ns as command_latency_ms,
    	e2e_latency.latency_ms as e2e_latency_ms
from ree_cloud_data_{environment}.telemetry
where
	ts='{ts}' and
	latitude <> 0 and longitude <> 0 and
	NOT ( ( (latitude < 52.45484564266689 -- drivery
        	AND latitude > 52.45220420838311
        	AND longitude < 13.389866352081299
        	AND longitude > 13.3833646774292) ) ) and 
	NOT ( ( (latitude < 52.39456322844031 -- Schönefeld
        	AND latitude > 52.3391150147957 
        	AND longitude < 13.557300567626951
        	AND longitude > 13.434391021728516 ) ) ) and
	NOT ( ( (latitude < 52.379607 -- Schonefeld 2
        	AND latitude > 52.375941
        	AND longitude < 13.564101
        	AND longitude > 13.557860) ) ) and
	NOT ( ( (latitude < 52.44473672286552 -- Marienfield
        	AND latitude > 52.43647000879358
        	AND longitude < 13.373408317565918
        	AND longitude > 13.364181518554688) ) )
        
order by timestamp_ms, session_id, vehicle_id
)
select 
    timebucket,
    session_id,
    vehicle_id,
    camera_index,
    
    avg(longitude) as longitude,
    avg(latitude) as latitude,

    avg(object_distance_m) as object_distance,
    avg(brake_pressure.brake_pressure) as brake_pressure,
    avg(lat_force) as force_lat,
    avg(lon_force) as force_lon,
    avg(yaw_rate_deg) as yaw_rate,
    avg(steering_wheel_deg) as steering_wheel,
    avg(steering_angle_deg) as steering_angle,
    avg(wheel_speed) as wheel_speed,
    avg(vehicle_ping_ms) as vehicle_ping,
    avg(rtp_lost) as rtp_lost,
    avg(rtp_late) as rtp_late,
    avg(modem0_rtt) as modem0_rtt,
    avg(modem1_rtt) as modem1_rtt,
    avg(modem2_rtt) as modem2_rtt,
    avg(modem3_rtt) as modem3_rtt,
    avg(modem0_rx) as modem0_rx,
    avg(modem1_rx) as modem1_rx,
    avg(modem2_rx) as modem2_rx,
    avg(modem3_rx) as modem3_rx,
    avg(modem0_tx) as modem0_tx,
    avg(modem1_tx) as modem1_tx,
    avg(modem2_tx) as modem2_tx,
    avg(modem3_tx) as modem3_tx,
    avg(camera_jitter_ms) as camera_jitter,
    avg(control_room_ram) as room_ram,
    avg(control_room_cpu) as room_cpu,
    avg(vehicle_ram) as vehicle_ram,
    avg(vehicle_cpu) as vehicle_cpu,

    max(camera_latency_ms) as camera_latency,
    max(command_latency_ms) as joystick_latency,
    max(e2e_latency_ms) as e2e_latency
    -- from_unixtime(timebucket)  as ts
    -- date_add('millisecond', (timebucket%100)*10 ,from_unixtime(timebucket/100))  as ts
from timebuckets
group by timebucket, session_id, vehicle_id, camera_index
order by timebucket
