--Of total revenue earned on each coin on a monthly basis, what percentage comes from reactivated customers?
--Reactivated - customers who were not active (no buy, sell, send, receive etc.) last month, but are active this month.


WITH first_buy_revenue AS (
    SELECT
        t.base_currency,
        SUM(revenue_usd) AS revenue_from_1st_buys,
        LEFT(transacted_at,7) AS month
    FROM experience_analytics.transactions t
    JOIN bitx_analytics.users_extended u ON t.user_id = u.id
    WHERE LEFT(u.first_buy_notified_at_2,7) = LEFT(t.transacted_at,7)
    AND t.base_currency IS NOT NULL
    GROUP BY t.base_currency, month
),
reactivated_revenue AS (
    SELECT
        t.base_currency,
        SUM(t.revenue_usd) AS revenue_from_reactivated,
        LEFT(t.transacted_at, 7) AS month
    FROM experience_analytics.transactions t
    LEFT JOIN (
        SELECT DISTINCT
            user_id
        FROM experience_analytics.transactions
        WHERE LEFT(transacted_at, 7) >= DATE_FORMAT(ADD_MONTHS(CURRENT_DATE(), -1), 'yyyy-MM-01')
        AND LEFT(transacted_at, 7) < DATE_FORMAT(CURRENT_DATE(), 'yyyy-MM-01')
    ) AS prev_month_users ON t.user_id = prev_month_users.user_id
    WHERE prev_month_users.user_id IS NULL
    AND t.base_currency IS NOT NULL
    GROUP BY t.base_currency, month

),
total_revenue AS (
    SELECT
        base_currency,
        SUM(revenue_usd) AS total_monthly_revenue,
        LEFT(transacted_at,7) AS month
    FROM experience_analytics.transactions
    WHERE base_currency IS NOT NULL
    GROUP BY base_currency, month
)
SELECT
    tr.base_currency,
    tr.month,
    fbr.revenue_from_1st_buys,
    rr.revenue_from_reactivated,
    tr.total_monthly_revenue,
    (fbr.revenue_from_1st_buys / tr.total_monthly_revenue) * 100 AS percentage_of_total_revenue_from_first_buys,
    (rr.revenue_from_reactivated / tr.total_monthly_revenue) * 100 AS percentage_of_total_revenue_from_reactivated
FROM total_revenue tr
LEFT JOIN first_buy_revenue fbr ON tr.base_currency = fbr.base_currency AND tr.month = fbr.month
LEFT JOIN reactivated_revenue rr ON tr.base_currency = rr.base_currency AND tr.month = rr.month
WHERE tr.month >= '2023-01'
AND tr.base_currency NOT IN ('ZAR','NGN','IDR')
ORDER BY tr.base_currency, tr.month;