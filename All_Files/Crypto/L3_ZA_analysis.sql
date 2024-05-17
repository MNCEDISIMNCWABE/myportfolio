SELECT made_first_buy, 
 kyc_usage_reason, 
 kyc_occupation_type_name, 
 kyc_source_of_funds,
 payment_method,
 COUNT(DISTINCT user_id) AS customer_count
FROM(

SELECT DISTINCT users.id AS user_id,
CASE WHEN (LEAST(
 users.first_transaction_names_at.buyada, users.first_transaction_names_at.tradebuy_ada,
 users.first_transaction_names_at.buyavax,
 users.first_transaction_names_at.buybch, users.first_transaction_names_at.tradebuy_bch,
 users.first_transaction_names_at.buybtc, users.first_transaction_names_at.tradebuy_btc,
 users.first_transaction_names_at.buyeth, users.first_transaction_names_at.tradebuy_eth,
 users.first_transaction_names_at.buylink, users.first_transaction_names_at.tradebuy_link,
 users.first_transaction_names_at.buyltc, users.first_transaction_names_at.tradebuy_ltc,
 users.first_transaction_names_at.buysol, users.first_transaction_names_at.tradebuy_sol,
 users.first_transaction_names_at.buyuni, users.first_transaction_names_at.tradebuy_uni,
 users.first_transaction_names_at.buyusdc, users.first_transaction_names_at.tradebuy_usdc,
 users.first_transaction_names_at.buyusdt,
 users.first_transaction_names_at.buyxrp, users.first_transaction_names_at.tradebuy_xrp
 )) IS NULL THEN "No" ELSE "Yes" 
END AS made_first_buy,
CASE
 WHEN kyc_usage_reasons.reason = 0 THEN 'Unknown'
 WHEN kyc_usage_reasons.reason = 1 THEN 'Investment'
 ELSE 'Unknown'
END AS kyc_usage_reason,
CASE
 WHEN kyc_occupations.occupation_type = 1 THEN 'Accounting'
 WHEN kyc_occupations.occupation_type = 2 THEN 'Auditing'
 ELSE 'Unknown'
END AS kyc_occupation_type_name,
CASE
 WHEN kyc_source_of_funds.source = 0 THEN 'Unknown'
 WHEN kyc_source_of_funds.source = 1 THEN 'Monthly Salary'
 ELSE NULL
END AS kyc_source_of_funds,
payments_extended.type_name AS payment_method
FROM bitx_analytics.users_extended AS users 
LEFT JOIN identity_analytics.kyc_usage_reasons ON kyc_usage_reasons.user_id = users.id
LEFT JOIN identity_analytics.kyc_occupations ON kyc_occupations.user_id = users.id
LEFT JOIN identity_analytics.kyc_source_of_funds ON kyc_source_of_funds.user_id = users.id
LEFT JOIN bitx_analytics.payments_extended ON payments_extended.user_id = users.id
WHERE users.verified_countries = ':ZA:'
AND users.kyc_level = 103)
GROUP BY 1,2,3,4,5