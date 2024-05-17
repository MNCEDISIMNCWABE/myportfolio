WITH revenue_summary AS (
    SELECT
        user_id,
        SUM(revenue_usd) AS total_revenue
    FROM
        analytics.experience.transactions
    GROUP BY
        user_id
),
revenue_rank AS (
    SELECT
        user_id,
        total_revenue,
        SUM(total_revenue) OVER (ORDER BY total_revenue DESC) AS cumulative_revenue,
        SUM(total_revenue) OVER () AS total_revenue_sum
    FROM
        revenue_summary
)
SELECT
    COUNT(user_id) AS num_customers
FROM
    revenue_rank
WHERE
    cumulative_revenue <= 0.8 * total_revenue_sum