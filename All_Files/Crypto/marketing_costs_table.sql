CREATE OR REPLACE VIEW analytics.bitx.vw_marketing_costs(
date COMMENT 'Denotes campaign date, referral or promo code applied date.',
cost_type COMMENT 'Denotes a marketing cost type such as referral, promo campaign or paid marketing/online advertising.',
cost_usd COMMENT 'Cost in USD associated with the marketing.'
) AS
SELECT
  date,
  cost_type,
  SUM(cost_usd) AS cost_usd
FROM (
  SELECT
    DATE(applied_at) AS date,
    COALESCE(reward_amount_usd, 0) AS cost_usd,
    'referral' AS cost_type
  FROM analytics.bitx.referrals
  UNION ALL
  SELECT
    DATE(applied_at) AS date,
    COALESCE(rr_reward_amount_usd, 0) AS cost_usd,
    'referrer' AS cost_type
  FROM analytics.bitx.referrers 
  UNION ALL
  SELECT
    DATE(created_at) AS date,
    COALESCE(amount_usd, 0) AS cost_usd,
    'promo' AS cost_type
  FROM analytics.bitx.promos 
  UNION ALL
  SELECT
    DATE(campaign_date) AS date,
    COALESCE(adn_cost, 0) AS cost_usd,
    'paid_marketing' AS cost_type
  FROM analytics.singular.vw_singular_campaigns
) consolidated_costs
GROUP BY date, cost_type;

ALTER VIEW analytics.bitx.vw_marketing_costs owner TO `schema_owners`;