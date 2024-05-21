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


--- V2 with user_id
CREATE OR REPLACE VIEW analytics.bitx.vw_marketing_costs AS
SELECT
  user_id,
  created_at,
  preferred_locale,
  provider,
  cost_type,
  base_cost_currency,
  SUM(cost) AS cost,
  SUM(cost_usd) AS cost_usd
FROM (
  SELECT
    r.user_id,
    DATE(r.applied_at) AS created_at,
    u.preferred_locale,
    'Luno' AS provider,
    'referral' AS cost_type,
    r.currency AS base_cost_currency,
    COALESCE(r.amount, 0) AS cost,
    COALESCE(r.reward_amount_usd, 0) AS cost_usd
  FROM analytics.bitx.referrals r
  LEFT JOIN analytics.bitx.users u
       ON r.user_id = u.id
  UNION ALL
  SELECT
    ref.user_id,
    DATE(ref.applied_at) AS created_at,
    u.preferred_locale,
    'Luno' AS provider,
    'referrer' AS cost_type,
    ref.rr_currency AS base_cost_currency,
    COALESCE(ref.rr_amount, 0) AS cost,
    COALESCE(ref.rr_reward_amount_usd, 0) AS cost_usd
  FROM analytics.bitx.referrers ref
  LEFT JOIN analytics.bitx.users u
       ON ref.user_id = u.id
  UNION ALL
  SELECT
    p.user_id,
    DATE(p.created_at) AS created_at,
    u.preferred_locale,
    'Luno' AS provider,
    'promo' AS cost_type,
    p.currency AS base_cost_currency,
    COALESCE(p.amount, 0) AS cost,
    COALESCE(p.amount_usd, 0) AS cost_usd
  FROM analytics.bitx.promos p
  LEFT JOIN analytics.bitx.users u
       ON p.user_id = u.id
  UNION ALL
  SELECT
    NULL AS user_id,
    DATE(campaign_date) AS created_at,
    country_field AS preferred_locale,
    'Luno' AS provider,
    'paid_marketing' AS cost_type,
    adn_original_currency AS base_cost_currency,
    COALESCE(adn_original_cost, 0) AS cost,
    COALESCE(adn_cost, 0) AS cost_usd
  FROM analytics.singular.vw_singular_campaigns
) consolidated_costs
GROUP BY user_id, created_at, preferred_locale, provider, cost_type, base_cost_currency;

ALTER VIEW analytics.bitx.vw_marketing_costs owner TO `schema_owners`;