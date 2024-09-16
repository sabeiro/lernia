with resample as (
    select session_id,
           timestamp_ms,
           'prod' as environment,
           e2e_latency.latency_ms as latency,
           FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1 * 1)) * (1 * 1) + (1 * 1))     as resampled_dt
    from ree_cloud_data_prod.telemetry  --    TABLESAMPLE BERNOULLI (45)
    union all
    select session_id,
           timestamp_ms,
           'stg' as environment,
           e2e_latency.latency_ms as latency,
           FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1 * 1)) * (1 * 1) + (1 * 1))     as resampled_dt
    from ree_cloud_data_stg.telemetry  --    TABLESAMPLE BERNOULLI (45)
),

     open_rows as (
         select
             session_id,
             resampled_dt,
             environment,
             latency,
             ROW_NUMBER() OVER
                 (
                 PARTITION BY session_id, resampled_dt
                 ORDER BY case when latency is null then 0 else 1 end desc, timestamp_ms DESC
                 ) AS Recency
         from resample r

     ),
     close_rows as (
         select
             session_id,
             resampled_dt,
             environment,
             latency,
             ROW_NUMBER() OVER
                 (
                 PARTITION BY session_id, resampled_dt
                 ORDER BY case when latency is null then 0 else 1 end desc, timestamp_ms ASC
                 ) AS Recency
         from resample r
     ),
     close as (
         select session_id, resampled_dt, latency, environment from close_rows
         where recency = 1
         order by session_id, resampled_dt, recency desc

     ),
     open as (
         select session_id, resampled_dt, latency, environment from open_rows
         where recency = 1
         order by session_id, resampled_dt, recency desc
     ),
     open_close as (
         select
            open.resampled_dt,
            open.session_id,
            open.environment,
            open.latency as open_e2e_latency,
            close.latency as close_e2e_latency
     from open inner join close on open.session_id=close.session_id and open.resampled_dt = close.resampled_dt
     ),
     prod as (
    select session_id,
           FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1 * 1)) * (1 * 1) + (1 * 1))     as resampled_dt,
           'prod' as environemnt,
           count(1) as messages,
           avg(coalesce(control_room_jitter.jitter_ms))                                                  as c_jitter_ms,
           avg(coalesce(vehicle_jitter.jitter_ms))                                                       as v_jitter_ms,
           avg(coalesce(control_room_ping.ping_ms))                                                      as c_ping_ms,
           avg(coalesce(vehicle_ping.ping_ms))                                                           as v_ping_ms,
           avg(coalesce(jitter.jitter_ms))                                                               as jitter_ms,
           avg(coalesce(joystick_hz.joystick_hz))                                                        as joystick_hz,
           avg(coalesce(steering_angle.angle_deg))                                                       as steering_angle,
           avg(coalesce(brake_pressure.brake_pressure))                                                  as brake_pressure,
           avg(coalesce(vehicle_disk_available.vehicle_disk_available_bytes) *
               0.00000095367432)                                                                         as disk_available_mb,
           avg(coalesce(longitude))                                                                      as longitude,
           avg(coalesce(latitude))                                                                       as latitude,
           avg(coalesce(e2e_latency.latency_ms))                                                         as e2e_latency,
           min(coalesce(e2e_latency.latency_ms))                                                         as min_e2e_latency,
           max(coalesce(e2e_latency.latency_ms))                                                         as max_e2e_latency,
           avg(coalesce(joystick_latency.joystick_latency_ns) / 1000)                                    as joystick_latency,
           avg(coalesce(wheel_speed.mean_km_per_hour))                                                   as mean_km_per_hour,
           avg(coalesce(vehicle_physics.lateral_force_m_per_sec_squared))                                as lateral_force_m_per_sec_squared,
           avg(coalesce(vehicle_physics.longitudinal_force_m_per_sec_squared))                           as longitudinal_force_m_per_sec_squared,
           avg(coalesce(vehicle_physics.yaw_rate_deg_per_sec))                                           as yaw_rate_deg_per_sec,
           avg(coalesce(control_room_ram_usage.ram_usage_percent))                                       as c_ram_usage_percent,
           avg(coalesce(vehicle_cpu_usage.cpu_usage_percent))                                            as v_cpu_usage_percent,
           avg(coalesce(vehicle_ram_usage.ram_usage_percent))                                            as v_ram_usage_percent
    from ree_cloud_data_prod.telemetry -- TABLESAMPLE BERNOULLI (5)
    where latitude <> 0
      and longitude <> 0
      and NOT (((latitude < 52.45484564266689
        AND latitude > 52.45220420838311
        AND longitude < 13.389866352081299
        AND longitude > 13.3833646774292)))
    group by 1, 2, 3
),
     stg as (
         select session_id,
                FROM_UNIXTIME((cast(to_unixtime(timestamp_ms) as INTEGER) / (1 * 1)) * (1 * 1) + (1 * 1))       as resampled_dt,
                'stg' as environemnt,
                count(1) as messages,
                avg(coalesce(control_room_jitter.jitter_ms))                                                  as c_jitter_ms,
                avg(coalesce(vehicle_jitter.jitter_ms))                                                       as v_jitter_ms,
                avg(coalesce(control_room_ping.ping_ms))                                                      as c_ping_ms,
                avg(coalesce(vehicle_ping.ping_ms))                                                           as v_ping_ms,
                avg(coalesce(jitter.jitter_ms))                                                               as jitter_ms,
                avg(coalesce(joystick_hz.joystick_hz))                                                        as joystick_hz,
                avg(coalesce(steering_angle.angle_deg))                                                       as steering_angle,
                avg(coalesce(brake_pressure.brake_pressure))                                                  as brake_pressure,
                avg(coalesce(vehicle_disk_available.vehicle_disk_available_bytes) *
                    0.00000095367432)                                                                         as disk_available_mb,
                avg(coalesce(longitude))                                                                      as longitude,
                avg(coalesce(latitude))                                                                       as latitude,
                avg(coalesce(e2e_latency.latency_ms))                                                         as e2e_latency,
                min(coalesce(e2e_latency.latency_ms))                                                         as min_e2e_latency,
                max(coalesce(e2e_latency.latency_ms))                                                         as max_e2e_latency,
                avg(coalesce(joystick_latency.joystick_latency_ns) / 1000)                                    as joystick_latency,
                avg(coalesce(wheel_speed.mean_km_per_hour))                                                   as mean_km_per_hour,
                avg(coalesce(vehicle_physics.lateral_force_m_per_sec_squared))                                as lateral_force_m_per_sec_squared,
                avg(coalesce(vehicle_physics.longitudinal_force_m_per_sec_squared))                           as longitudinal_force_m_per_sec_squared,
                avg(coalesce(vehicle_physics.yaw_rate_deg_per_sec))                                           as yaw_rate_deg_per_sec,
                avg(coalesce(control_room_ram_usage.ram_usage_percent))                                       as c_ram_usage_percent,
                avg(coalesce(vehicle_cpu_usage.cpu_usage_percent))                                            as v_cpu_usage_percent,
                avg(coalesce(vehicle_ram_usage.ram_usage_percent))                                            as v_ram_usage_percent
         from ree_cloud_data_stg.telemetry -- TABLESAMPLE BERNOULLI (5)
         where latitude <> 0
           and longitude <> 0
           and NOT (((latitude < 52.45484564266689
             AND latitude > 52.45220420838311
             AND longitude < 13.389866352081299
             AND longitude > 13.3833646774292)))
         group by 1, 2, 3
     ),
     prod_stg as (
         select *
         from prod
         union all
         select *
         from stg
         order by 2
     )
select p.*,
       oc.open_e2e_latency,
       oc.close_e2e_latency
from prod_stg p
join open_close oc
on prod_stg.resampled_dt = open_close.resampled_dt
and prod_stg.session_id = open_close.session_id
and prod_stg.environemnt = open_close.environment
