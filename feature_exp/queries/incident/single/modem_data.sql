-- modem_round_trip_snapshot 
-- SELECT
--   (cast(to_unixtime(timestamp_ms) * 1000 as bigint) / 100) * 100 timechunk,
--   avg(modem_report.modem_report_data[1].tunnel_rtt) AS mr0rtt_avg,
--   avg(modem_report.modem_report_data[2].tunnel_rtt) AS mr1rtt_avg,
--   avg(modem_report.modem_report_data[3].tunnel_rtt) AS mr2rtt_avg,
--   avg(modem_report.modem_report_data[4].tunnel_rtt) AS mr3rtt_avg
-- FROM ree_cloud_data_{{ environment }}.telemetry
-- WHERE cast(to_unixtime(timestamp_ms) * 1000 as bigint) BETWEEN {{snapshot}} - (1000000 * 10) AND {{snapshot}} + (1000000 * 10)
-- AND session_id = '{{ session_id }}'
-- GROUP BY 1 ORDER BY 1 ASC LIMIT 10000
with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
)
select
   to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
  modem_report.modem_report_data[1].tunnel_rtt AS mr0rtt,
  modem_report.modem_report_data[2].tunnel_rtt AS mr1rtt,
  modem_report.modem_report_data[3].tunnel_rtt AS mr2rtt,
  modem_report.modem_report_data[4].tunnel_rtt AS mr3rtt
from ree_cloud_data_{{ environment }}.telemetry --TABLESAMPLE BERNOULLI (10)
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and modem_report is not null
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
