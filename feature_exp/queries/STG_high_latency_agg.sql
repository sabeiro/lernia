WITH PARAMETERS AS (
    SELECT

            to_unixtime(current_timestamp - interval '28' day) * 1000 as start_time,
            to_unixtime(now()) * 1000 as end_time,
            400 delay
),
     partitions_old AS (
         SELECT
             DISTINCT date_partition
         FROM ree_cloud_persist_stg.telemetry_data as t
                  RIGHT JOIN PARAMETERS p ON date_partition BETWEEN

             date_format(
                     from_unixtime(p.start_time / 1000),
                     '%Y-%m-%d-%H'
                 )
             AND date_format(
                     from_unixtime(p.end_time / 1000),
                     '%Y-%m-%d-%H'
                 )
         order by
             date_partition
     )
        ,
     partitions_new AS (
         SELECT
             DISTINCT dt
         FROM "ree_cloud_data_stg"."telemetry" as t
                  RIGHT JOIN PARAMETERS p ON dt BETWEEN
             date_format(
                     from_unixtime(p.start_time / 1000),
                     '%Y-%m-%d'
                 )
             AND date_format(
                     from_unixtime(p.end_time / 1000),
                     '%Y-%m-%d'
                 )
         order by
             dt
     ),
     EVENTS_old AS (
         SELECT
             p.date_partition,
             p.latitude,
             p.longitude,
             p.session_id,
             from_unixtime(p.timestamp_ms / 1000 ) as timestamp_ms,
             p.e2e_latency.latency_ms as latency,
             CASE
                 WHEN p.e2e_latency.latency_ms > (
                     SELECT
                         delay
                     FROM
                         PARAMETERS
                 ) THEN TRUE
                 ELSE FALSE
                 END is_delay
         FROM
             ree_cloud_persist_stg.telemetry_data p TABLESAMPLE BERNOULLI (1)
                 right join partitions_old on p.date_partition = partitions_old.date_partition
         WHERE
             p.e2e_latency.latency_ms IS NOT NULL
           and p.timestamp_ms IS NOT NULL
           and p.timestamp_ms > 0
         order by
             session_id,
             timestamp_ms
     ),
     EVENTS_new AS (
         SELECT
             p.dt as date_partition,
             p.latitude,
             p.longitude,
             p.session_id,
             p.timestamp_ms,
             p.e2e_latency.latency_ms as latency,
             CASE
                 WHEN p.e2e_latency.latency_ms > (
                     SELECT
                         delay
                     FROM
                         PARAMETERS
                 ) THEN TRUE
                 ELSE FALSE
                 END is_delay
         FROM
             ree_cloud_data_stg.telemetry p TABLESAMPLE BERNOULLI (1)
                 right join partitions_new on p.dt = partitions_new.dt
         WHERE
             p.e2e_latency.latency_ms IS NOT NULL
           and p.timestamp_ms IS NOT NULL
         order by
             session_id,
             timestamp_m
     ),

     events as (
         select * from EVENTS_old
         union all
         select * from EVENTS_new
     ),
    all_events_loc_filters as (
        select * from events
        where
            latitude <> 0 and longitude <> 0 and
            case when (select filter_lat_lon from parameters) = True
                then
                     not (
                              (
                                  (latitude < 52.45484564266689 and latitude > 52.45220420838311 and longitude < 13.389866352081299 and longitude > 13.3833646774292) or  -- Drivery
                                  (latitude < 52.39456322844031 and latitude > 52.3391150147957 and longitude < 13.557300567626951 and longitude > 13.434391021728516)    -- Schonefeld
                              )
                          )
             end

            ),


     window1 as (
         select
             session_id,
             timestamp_ms,
             latency,
             is_delay,
             min(timestamp_ms) over(partition by session_id, is_delay) as start_ts,
             max(latency) over(partition by session_id, is_delay) as e2e_latency_max,
             max(timestamp_ms) over(partition by session_id, is_delay) as end_ts
         from
             all_events_loc_filters
     )

select
    session_id,
    date_format(start_ts, '%Y-%m-%d %k:%i') as start_ts,
    arbitrary(is_delay) as is_delay,
    date_diff('second', start_ts, arbitrary(end_ts)) as duration,
    arbitrary(e2e_latency_max) as e2e_latency_max
from
    window1
where
    is_delay
group by
    session_id,
    start_ts
