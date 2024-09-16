SELECT
	latitude,
    	longitude,
    	session_id,
    	timestamp_ms,
    	latency_camera.latency_ms as latency
FROM
	ree_cloud_data_prod.telemetry
WHERE
	latency_camera.latency_ms IS NOT NULL
    	and timestamp_ms IS NOT NULL
    	and latitude <> 0
    	and longitude <> 0
    	and not (
        (latitude < 52.45484564266689 and latitude > 52.45220420838311 and longitude < 13.389866352081299 and longitude > 13.3833646774292)  -- Drivery
	    or  
        (latitude < 52.39456322844031 and latitude > 52.3391150147957 and longitude < 13.557300567626951 and longitude > 13.434391021728516) -- Schonefeld
    	)
	and dt = '2020-07-08'
order by
        session_id,
        timestamp_ms
