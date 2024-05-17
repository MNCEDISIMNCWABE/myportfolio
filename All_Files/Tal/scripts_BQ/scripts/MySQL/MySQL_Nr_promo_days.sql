SELECT prod.product_id, prod.plid, x.* 
FROM (
SELECT 
    p.id, p.group_id, p.name, 
    p.display_name, 
    DATEDIFF(p.date_end,p.date_start) as num_promo_days,
    DATE(p.date_start) AS date_start, 
    DATE(p.date_end) AS date_end, 
    date_format(p.date_start,"%Y-%m") AS YearMonth,
    YEAR(p.date_start) AS yr
FROM deals.promotions p
WHERE p.name NOT LIKE "%DD-daily-deals%"
     AND date_start>= '2021-01-01' AND date_start <'2021-04-01'
UNION ALL
SELECT 
   p.id, p.group_id, p.name, 
   p.display_name, 
   DATEDIFF(p.date_end,p.date_start) as num_promo_days,
   DATE(p.date_start) AS date_start, 
  DATE(p.date_end) AS date_end, 
  date_format(p.date_start,"%Y-%m") AS YearMonth,
  YEAR(p.date_start) AS yr
FROM deals.promotions p
WHERE p.name NOT LIKE "%DD-daily-deals%"
     AND date_start>= '2022-01-01' AND date_start <'2022-04-01') x
INNER JOIN deals.promotion_products prod 
  ON prod.promotion_id = x.id
  GROUP BY product_id, plid