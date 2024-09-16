with buckets as (select
    timestamp_ms,
    --cast(cast(to_unixtime(timestamp_ms)*1000 as bigint) as varchar) as snapshot,
    cast(to_unixtime(timestamp_ms)*1000 as bigint)/5000*5000 as bucket,
    session_id,
    fault_detected.fault_source
from "ree_cloud_views_prod"."fault_detected"
where
    actuators_enabled and nebra and fault_detected.fault_source != 'FAULT_EMERGENCY_BUTTON'
order by timestamp_ms desc)
select
	session_id,
	cast(cast(to_unixtime(min(timestamp_ms))*1000 as bigint) as varchar) as snapshot,
	min(timestamp_ms) as timestamp_ms,
	array_agg(fault_source) as fault_source
from buckets
group by session_id, bucket
order by timestamp_ms desc
