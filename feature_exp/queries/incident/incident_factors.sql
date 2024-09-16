with ts_range as (
select distinct ts
from ree_cloud_data_{environment}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({snapshot}/1000)), '%Y-%m-%d')
    and session_id = '{session_id}'
), timebuckets as (
select
	timestamp_ms,
	floor(to_unixtime(timestamp_ms)*10)/10 as timebucket,
	
	radar_track.object_distance_m as object_distance_m,
	brake_pressure,
	sum(case when steering_wheel_state is null then 0 else 1 end) as steering_wheel_state_partition,
	sum(case when brake_pressure is null then 0 else 1 end) as brake_pressure_partition,
	-- sum(case when steering_wheel_state is null then 0 else 1 end) over (order by timestamp_ms) as steering_wheel_state_partition,
	-- sum(case when brake_pressure is null then 0 else 1 end) over (order by timestamp_ms) as brake_pressure_partition,
	vehicle_physics.lateral_force_m_per_sec_squared as lat_force,
	vehicle_physics.longitudinal_force_m_per_sec_squared as lon_force,
	-1.0 * vehicle_physics.yaw_rate_deg_per_sec as yaw_rate_deg,
	steering_wheel_state.steering_deg steering_wheel_deg,
    	steering_angle.angle_deg steering_angle_deg,
	-- steering_wheel_state.brake / -655.35 as td_brake,
	-- steering_wheel_state.throttle / 655.35 as td_throttle,
	-- steering_interval_max.interval_ms as steering_interval,
	wheel_speed.mean_km_per_hour as wheel_speed,
	vehicle_ping.ping_ms as vehicle_ping_ms,
	rtp_stats.lost as rtp_lost,
   	rtp_stats.late as rtp_late,
	modem_report.modem_report_data[1].tunnel_delta_rx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[1].tunnel_delta_interval_ns / 1000000000.0)) AS modem0_rx,
	modem_report.modem_report_data[1].tunnel_delta_rx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[2].tunnel_delta_interval_ns / 1000000000.0)) AS modem1_rx,
	modem_report.modem_report_data[1].tunnel_delta_rx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[3].tunnel_delta_interval_ns / 1000000000.0)) AS modem2_rx,
	modem_report.modem_report_data[1].tunnel_delta_rx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[4].tunnel_delta_interval_ns / 1000000000.0)) AS modem3_rx,
	modem_report.modem_report_data[1].tunnel_delta_tx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[1].tunnel_delta_interval_ns / 1000000000.0)) AS modem0_tx,
	modem_report.modem_report_data[1].tunnel_delta_tx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[2].tunnel_delta_interval_ns / 1000000000.0)) AS modem1_tx,
	modem_report.modem_report_data[1].tunnel_delta_tx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[3].tunnel_delta_interval_ns / 1000000000.0)) AS modem2_tx,
	modem_report.modem_report_data[1].tunnel_delta_tx / (modem_report.modem_report_data[1].tunnel_delta_interval_sec + (modem_report.modem_report_data[4].tunnel_delta_interval_ns / 1000000000.0)) AS modem3_tx,
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
    	(joystick_latency.joystick_latency_ns/1000000) as command_latency_ms,
    	e2e_latency.latency_ms as e2e_latency_ms
from ree_cloud_data_{environment}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{session_id}'
    -- and (latency_camera is not null or joystick_latency is not null or e2e_latency is not null)
    and abs({snapshot}/1000.0 - to_unixtime(timestamp_ms)) < {time_horizon_sec}
order by timestamp_ms
)
select 
    timebucket,
    timebucket*1000 - {snapshot} as ts,

    avg(object_distance_m) as object_distance,
    avg(brake_pressure.brake_pressure) as brake_pressure,
    max(brake_pressure_partition) as brake_partition,
    max(steering_wheel_state_partition) as steering_partition,
    avg(lat_force) as force_lat,
    avg(lon_force) as force_lon,
    avg(yaw_rate_deg) as yaw_rate,
    avg(steering_wheel_deg)/32767.0 * 420.0 as steering_wheel,
    avg(steering_angle_deg) as steering_angle,
    avg(td_brake) as td_brake,
    -- avg(td_throttle) as td_throttle,
    -- avg(wheel_speed) as wheel_speed,
    -- avg(steering_interval) as steering_jitter,
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
group by timebucket
order by timebucket
