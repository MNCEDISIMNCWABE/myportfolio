-- Create a consolidated marketing costs table

CREATE OR REPLACE TABLE analytics.bitx.marketing_costs AS
SELECT
  user_id,
  SUM(COALESCE(cost_usd, 0)) AS cost_usd,
  applied_at AS date,
  cost_type
FROM (
  -- Costs from referrals
  SELECT
    user_id,
    COALESCE(reward_amount_usd, 0) AS cost_usd,
    applied_at,
    'referral' AS cost_type
  FROM analytics.bitx.referrals
  UNION ALL
  -- Costs from referrers
  SELECT
    user_id,
    COALESCE(rr_reward_amount_usd, 0) AS cost_usd,
    applied_at,
    'referrer' AS cost_type
  FROM analytics.bitx.referrers
  UNION ALL
  -- Costs from promos
  SELECT
    user_id,
    COALESCE(amount_usd, 0) AS cost_usd,
    created_at AS applied_at,
    'promo' AS cost_type
  FROM analytics.bitx.promos
) consolidated_costs
GROUP BY
  user_id,
  applied_at,
  cost_type
ORDER BY
  user_id,
  applied_at;
