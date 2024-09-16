with ts_range as (
select distinct ts
from ree_cloud_data_{environment}.telemetry
where 
	ts='{ts}'
), timebuckets as (
select
	timestamp_ms,
	-- floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
	floor(to_unixtime(timestamp_ms)) as timebucket,
	
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

	modem_report.modem_report_data[1].rssi AS modem0_rssi,
	modem_report.modem_report_data[2].rssi AS modem1_rssi,
	modem_report.modem_report_data[3].rssi AS modem2_rssi,
	modem_report.modem_report_data[4].rssi AS modem3_rssi,
	modem_report.modem_report_data[1].sinr AS modem0_sinr,
	modem_report.modem_report_data[2].sinr AS modem1_sinr,
	modem_report.modem_report_data[3].sinr AS modem2_sinr,
	modem_report.modem_report_data[4].sinr AS modem3_sinr,
	modem_report.modem_report_data[1].rsrp AS modem0_rsrp,
	modem_report.modem_report_data[2].rsrp AS modem1_rsrp,
	modem_report.modem_report_data[3].rsrp AS modem2_rsrp,
	modem_report.modem_report_data[4].rsrp AS modem3_rsrp,
	modem_report.modem_report_data[1].rsrq AS modem0_rsrq,
	modem_report.modem_report_data[2].rsrq AS modem1_rsrq,
	modem_report.modem_report_data[3].rsrq AS modem2_rsrq,
	modem_report.modem_report_data[4].rsrq AS modem3_rsrq,
	modem_report.modem_report_data[1].cell_id AS modem0_cell,
	modem_report.modem_report_data[2].cell_id AS modem1_cell,
	modem_report.modem_report_data[3].cell_id AS modem2_cell,
	modem_report.modem_report_data[4].cell_id AS modem3_cell,
	
	jitter.jitter_ms as camera_jitter_ms,
	control_room_ram_usage.ram_usage_percent as control_room_ram,
	control_room_cpu_usage.cpu_usage_percent as control_room_cpu,
  	vehicle_ram_usage.ram_usage_percent as vehicle_ram,
  	vehicle_cpu_usage.cpu_usage_percent as vehicle_cpu,

	latency_camera.latency_ms as camera_latency_ms,
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
    
    avg(modem0_rssi) as modem0_rssi,
    avg(modem1_rssi) as modem1_rssi,
    avg(modem2_rssi) as modem2_rssi,
    avg(modem3_rssi) as modem3_rssi,
    avg(modem0_sinr) as modem0_sinr,
    avg(modem1_sinr) as modem1_sinr,
    avg(modem2_sinr) as modem2_sinr,
    avg(modem3_sinr) as modem3_sinr,
    avg(modem0_rsrp) as modem0_rsrp,
    avg(modem1_rsrp) as modem1_rsrp,
    avg(modem2_rsrp) as modem2_rsrp,
    avg(modem3_rsrp) as modem3_rsrp,
    avg(modem0_rsrq) as modem0_rsrq,
    avg(modem1_rsrq) as modem1_rsrq,
    avg(modem2_rsrq) as modem2_rsrq,
    avg(modem3_rsrq) as modem3_rsrq,
    arbitrary(modem0_cell) as modem0_cell,
    arbitrary(modem1_cell) as modem1_cell,
    arbitrary(modem2_cell) as modem2_cell,
    arbitrary(modem3_cell) as modem3_cell,
    
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
group by timebucket, session_id, vehicle_id
order by timebucket
