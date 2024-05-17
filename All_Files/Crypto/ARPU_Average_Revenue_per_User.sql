---Month cohorts
WITH FirstTransaction AS (
    SELECT user_id, MIN(DATE(transacted_at)) AS first_transaction_date
    FROM experience_analytics.transactions
    GROUP BY user_id
),
DateRange AS ( 
    SELECT user_id, EXPLODE(SEQUENCE(first_transaction_date, date_add(first_transaction_date, 90), interval 1 day)) AS transacted_at
    FROM FirstTransaction
), 
TransactionsWithRevenue AS (
    SELECT 
        t.user_id,
        DATE(t.transacted_at) AS transacted_at_date,
        SUM(t.revenue_usd) AS revenue
    FROM experience_analytics.transactions t
    WHERE t.transaction_name IS NOT NULL
    GROUP BY t.user_id, DATE(t.transacted_at)
),
RevenueWithAllDays AS (
    SELECT 
        dr.user_id,
        ft.first_transaction_date,  
        dr.transacted_at as transacted_at,
        DATEDIFF(dr.transacted_at, ft.first_transaction_date) AS days_since_first_transaction,
        IFNULL(twr.revenue, 0) AS revenue
    FROM DateRange dr
    LEFT JOIN TransactionsWithRevenue twr ON dr.user_id = twr.user_id AND dr.transacted_at = twr.transacted_at_date
    JOIN FirstTransaction ft ON dr.user_id = ft.user_id
),
CumulativeRevenue AS (
    SELECT 
        user_id,
        first_transaction_date,
        transacted_at,
        days_since_first_transaction,
        revenue,
        SUM(revenue) OVER (PARTITION BY user_id ORDER BY transacted_at) AS cumulative_revenue
    FROM RevenueWithAllDays
    WHERE transacted_at <= CURRENT_DATE 
)

SELECT 
    days_since_first_transaction,

    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-01' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Jan2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-02' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Feb2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-03' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Mar2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-04' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Apr2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-05' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS May2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-06' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Jun2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-07' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Jul2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-08' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Aug2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-09' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Sep2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-10' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Oct2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-11' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Nov2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2023-12' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Dec2023,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2024-01' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Jan2024,
    AVG(CASE WHEN LEFT(first_transaction_date,7) = '2024-02' AND days_since_first_transaction <= DATEDIFF(CURRENT_DATE, first_transaction_date) THEN cumulative_revenue ELSE NULL END) AS Feb2024
FROM CumulativeRevenue
GROUP BY 1
ORDER BY days_since_first_transaction;
