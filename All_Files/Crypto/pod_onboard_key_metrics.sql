SELECT 
 DATE(DATE_FORMAT(users.email_verified_at,'yyy-MM-dd')) AS users_dynamic_email_verified_date_formatted,
 DATE(DATE_FORMAT(users.created_at,'yyy-MM-dd')) AS users_dynamic_signup_date_formatted,
 DATE(DATE_FORMAT(users.phone_first_verified_at,'yyy-MM-dd')) AS users_dynamic_phone_verified_date_formatted,
 user_primary_fiat_currencies.primary_fiat_currency AS user_primary_fiat_currencies_primary_fiat_currency,
 CASE
 WHEN users.signup_client_id IS NULL OR users.signup_client_id = 'bitx_web' OR users.signup_client_id = '' THEN 'Web'
 WHEN users.signup_client_id = 'bitx_ios' THEN 'iOS'
 WHEN users.signup_client_id = 'bitx_android' THEN 'Android'
 END AS users_signup_platform,
 CASE 
 WHEN demographics.gender = 1 THEN 'Female'
 WHEN demographics.gender = 2 THEN 'Male' ELSE 'Unknown' 
 END AS gender,
 COUNT(DISTINCT CASE WHEN (((DATE_FORMAT(users.email_verified_at, 'yyyy-MM-dd')) IS NULL AND ((CAST(users.created_at AS VARCHAR(29))) >= '2018-12-02 22:00:00' AND (CAST(users.created_at AS VARCHAR(29))) < '2018-12-03 06:00:00')) = 'FALSE') THEN users.id ELSE NULL END) AS users_number_of_unique_customers,
 COUNT(DISTINCT CASE WHEN ((users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.created_at)) >= 0 AND (users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.created_at)) < 7*24*60*60) AND users.email_verified_at IS NOT NULL AND users.phone_first_verified_at IS NOT NULL THEN users.id ELSE NULL END) AS users_count_signup_to_kyc_7d,
 (COUNT(DISTINCT CASE WHEN ((users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.created_at)) >= 0 AND (users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.created_at)) < 7*24*60*60) AND users.email_verified_at IS NOT NULL AND users.phone_first_verified_at IS NOT NULL THEN users.id ELSE NULL END)/COUNT(DISTINCT CASE WHEN (((DATE_FORMAT(users.email_verified_at, 'yyyy-MM-dd')) IS NULL AND ((CAST(users.created_at AS VARCHAR(29))) >= '2018-12-02 22:00:00' AND (CAST(users.created_at AS VARCHAR(29))) < '2018-12-03 06:00:00')) = 'FALSE') THEN users.id ELSE NULL END))*100 AS perc_signup_to_kyc_7d,
 COUNT(DISTINCT CASE WHEN ((UNIX_TIMESTAMP(users.email_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 86400 >= 0 AND (UNIX_TIMESTAMP(users.email_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 86400 < 7) AND users.email_verified_at IS NOT NULL THEN users.id ELSE NULL END) AS users_count_signup_to_email_verified_7d,
 (COUNT(DISTINCT CASE WHEN ((UNIX_TIMESTAMP(users.email_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 86400 >= 0 AND (UNIX_TIMESTAMP(users.email_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 86400 < 7) AND users.email_verified_at IS NOT NULL THEN users.id ELSE NULL END)/COUNT(DISTINCT CASE WHEN (((DATE_FORMAT(users.email_verified_at, 'yyyy-MM-dd')) IS NULL AND ((CAST(users.created_at AS VARCHAR(29))) >= '2018-12-02 22:00:00' AND (CAST(users.created_at AS VARCHAR(29))) < '2018-12-03 06:00:00')) = 'FALSE') THEN users.id ELSE NULL END))*100 AS perc_signup_to_email_verified_7d,
 COUNT(DISTINCT CASE WHEN ((UNIX_TIMESTAMP(users.phone_first_verified_at) - UNIX_TIMESTAMP(users.email_verified_at)) / 86400 >= 0 AND (UNIX_TIMESTAMP(users.phone_first_verified_at) - UNIX_TIMESTAMP(users.email_verified_at)) / 86400 < 7) AND users.email_verified_at IS NOT NULL AND users.phone_first_verified_at IS NOT NULL THEN users.id ELSE NULL END) AS users_count_email_verified_to_phone_verified_7d,
 (COUNT(DISTINCT CASE WHEN ((UNIX_TIMESTAMP(users.phone_first_verified_at) - UNIX_TIMESTAMP(users.email_verified_at)) / 86400 >= 0 AND (UNIX_TIMESTAMP(users.phone_first_verified_at) - UNIX_TIMESTAMP(users.email_verified_at)) / 86400 < 7) AND users.email_verified_at IS NOT NULL AND users.phone_first_verified_at IS NOT NULL THEN users.id ELSE NULL END)/ COUNT(DISTINCT CASE WHEN (((DATE_FORMAT(users.email_verified_at, 'yyyy-MM-dd')) IS NULL AND ((CAST(users.created_at AS VARCHAR(29))) >= '2018-12-02 22:00:00' AND (CAST(users.created_at AS VARCHAR(29))) < '2018-12-03 06:00:00')) = 'FALSE') THEN users.id ELSE NULL END))*100 AS perc_email_verified_to_phone_verified_7d,
 COUNT(DISTINCT CASE WHEN ((users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.phone_first_verified_at)) >= 0 AND (users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.phone_first_verified_at)) < 7*24*60*60) AND users.email_verified_at IS NOT NULL THEN users.id ELSE NULL END) AS users_count_phone_verified_to_kyc_7d,
 (COUNT(DISTINCT CASE WHEN ((users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.phone_first_verified_at)) >= 0 AND (users.identity_verified_timestamp/1000 - UNIX_TIMESTAMP(users.phone_first_verified_at)) < 7*24*60*60) AND users.email_verified_at IS NOT NULL THEN users.id ELSE NULL END)/ COUNT(DISTINCT CASE WHEN (((DATE_FORMAT(users.email_verified_at, 'yyyy-MM-dd')) IS NULL AND ((CAST(users.created_at AS VARCHAR(29))) >= '2018-12-02 22:00:00' AND (CAST(users.created_at AS VARCHAR(29))) < '2018-12-03 06:00:00')) = 'FALSE') THEN users.id ELSE NULL END))*100 AS perc_phone_kyc_7d,
 
 
 
 MEDIAN (CASE
 WHEN users.kyc_level = 103 AND
 (
 (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60 >= 0
 AND (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60 <= 6*24*60
 )
 THEN
 (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60
 ELSE
 NULL
 END) AS users_signup_to_kyc_l3_median_in_mins,

 PERCENTILE_APPROX(
 CASE
 WHEN users.kyc_level = 103 AND
 (
 (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60 >= 0
 AND (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60 <= 6*24*60
 )
 THEN
 (UNIX_TIMESTAMP(users.kyc_level_3_verified_at) - UNIX_TIMESTAMP(users.created_at)) / 60
 ELSE
 NULL
 END,
 0.25 
 ) AS users_signup_to_kyc_l3_25th_percentile_in_mins

FROM 
 bitx_analytics.users_extended AS users
LEFT JOIN 
 bitx_analytics.user_primary_fiat_currencies ON user_primary_fiat_currencies.user_id = users.id
LEFT JOIN 
 bitx_analytics.vw_demographics AS demographics ON users.id = demographics.user_id

WHERE 
 users.created_at >= DATE_ADD(CURRENT_TIMESTAMP(), -90)
GROUP BY 1,2,3,4,5,6