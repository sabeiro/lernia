select session_id,
       vehicle_id,
       FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1*1))*(1*1)+(1*1)) as resampled_dt,
       avg(longitude) as lat,
       avg(latitude) as lon,
       max(actuators_enabled) as actuator_enabled,
       avg(e2e_latency.latency_ms) as e2e_latency,
       avg(latency_camera.latency_ms) as camera_latency,
       avg(joystick_latency.joystick_latency_ns) as joystick_latency,
       avg(wheel_speed.mean_km_per_hour) as mean_km_per_hour,
       avg(vehicle_physics.lateral_force_m_per_sec_squared) as lateral_force_m_per_sec_squared,
       avg(vehicle_physics.longitudinal_force_m_per_sec_squared) as longitudinal_force_m_per_sec_squared,
       avg(vehicle_physics.yaw_rate_deg_per_sec) as yaw_rate_deg_per_sec,
       avg(control_room_ram_usage.ram_usage_percent) as c_ram_usage_percent,
       avg(vehicle_cpu_usage.cpu_usage_percent) as v_cpu_usage_percent,
       avg(vehicle_ram_usage.ram_usage_percent) as v_ram_usage_percent
from ree_cloud_data_prod.telemetry
where
    latitude <> 0 and longitude <> 0 and
    NOT ( ( (latitude < 52.45484564266689
        AND latitude > 52.45220420838311
        AND longitude < 13.389866352081299
        AND longitude > 13.3833646774292) ) )
	and year='2020' and month='07' and day='08'
group by session_id, vehicle_id, FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1*1))*(1*1)+(1*1))
order by FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1*1))*(1*1)+(1*1))
