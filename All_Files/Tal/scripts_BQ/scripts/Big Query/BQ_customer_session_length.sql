SELECT SUM(session_length) AS session_dur,
COUNT (client_id) AS num_sessions,
activity_type,
case when DATE(session_start)>='2021-09-01' and DATE(session_start)<'2021-09-03'  then '1 week'
when DATE(session_start)>='2021-09-03' and DATE(session_start)<'2021-09-06'  then '1 weekend'
when DATE(session_start)>='2021-09-06' and DATE(session_start)<'2021-09-10'  then '1 week'
when DATE(session_start)>='2021-09-10' and DATE(session_start)<'2021-09-13'  then '2 weekend'
when DATE(session_start)='2021-09-13'  and DATE(session_start)<'2021-09-17'  then '2 week'
when DATE(session_start)>='2021-09-17' and DATE(session_start)<'2021-09-20'  then '3 weekend'
when DATE(session_start)>='2021-09-20' and DATE(session_start)<'2021-09-24'  then '3 week'
when DATE(session_start)>='2021-09-24' and DATE(session_start)<'2021-09-27'  then '4 weekend'
when DATE(session_start)>='2021-09-27' and DATE(session_start)<'2021-10-01'  then '4 week'
when DATE(session_start)>='2021-10-01' and DATE(session_start)<'2021-10-04'  then '1 weekend'
when DATE(session_start)>='2021-10-04' and DATE(session_start)<'2021-10-08'  then '1 week'
when DATE(session_start)>='2021-10-08' and DATE(session_start)<'2021-10-11'  then '1 weekend'
when DATE(session_start)>='2021-10-11' and DATE(session_start)<'2021-10-15'  then '2 week'
when DATE(session_start)>='2021-10-15' and DATE(session_start)<'2021-10-18'  then '2 weekend'
when DATE(session_start)>='2021-10-18' and DATE(session_start)<'2021-10-22'  then '3 week'
when DATE(session_start)>='2021-10-22' and DATE(session_start)<'2021-10-25'  then '3 weekend'
when DATE(session_start)>='2021-10-25' and DATE(session_start)<'2021-10-29'  then '4 week'
when DATE(session_start)>='2021-10-29' and DATE(session_start)<'2021-11-01'  then '4 weekend'
end as Week_Weekend
FROM `gcp-takealot.prod_aggregations.client_session_aggregation`,
UNNEST(session_data)
WHERE DATE(session_start) >= "2021-09-01" AND DATE(session_start) < "2021-11-01" 
AND activity_type LIKE "%click_through%"
AND  activity_type NOT IN ('open_screen')
AND is_bot = 0
GROUP BY Week_Weekend, activity_type