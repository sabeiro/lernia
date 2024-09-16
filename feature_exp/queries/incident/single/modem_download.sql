with ts_range as (
select distinct ts
from ree_cloud_data_{{ environment }}.telemetry
where 
    dt = date_format(date_trunc('day', from_unixtime({{snapshot}}/1000)), '%Y-%m-%d')
    and session_id = '{{ session_id }}'
)
select
   to_unixtime(timestamp_ms)*1000 - {{snapshot}} as ts,
   modem_report.modem_report_data[1].tunnel_delta_rx / 
      (modem_report.modem_report_data[1].tunnel_delta_interval_sec + 
      (modem_report.modem_report_data[1].tunnel_delta_interval_ns / 1000000000.0)) AS m0_rx_per_sec,
   modem_report.modem_report_data[2].tunnel_delta_rx / 
      (modem_report.modem_report_data[2].tunnel_delta_interval_sec + 
      (modem_report.modem_report_data[2].tunnel_delta_interval_ns / 1000000000.0)) AS m1_rx_per_sec,
   modem_report.modem_report_data[3].tunnel_delta_rx / 
      (modem_report.modem_report_data[3].tunnel_delta_interval_sec + 
      (modem_report.modem_report_data[3].tunnel_delta_interval_ns / 1000000000.0)) AS m2_rx_per_sec,
   modem_report.modem_report_data[4].tunnel_delta_rx / 
      (modem_report.modem_report_data[4].tunnel_delta_interval_sec + 
      (modem_report.modem_report_data[4].tunnel_delta_interval_ns / 1000000000.0)) AS m3_rx_per_sec
from ree_cloud_data_{{ environment }}.telemetry
where
    ts in (select ts from ts_range)
    and session_id = '{{ session_id }}'
    and modem_report is not null
    and (modem_report.modem_report_data[1].tunnel_delta_valid = true
    or modem_report.modem_report_data[2].tunnel_delta_valid = true
    or modem_report.modem_report_data[3].tunnel_delta_valid = true
    or modem_report.modem_report_data[4].tunnel_delta_valid = true)
    and abs({{snapshot}}/1000.0 - to_unixtime(timestamp_ms)) < {{time_horizon_sec}}
order by timestamp_ms
