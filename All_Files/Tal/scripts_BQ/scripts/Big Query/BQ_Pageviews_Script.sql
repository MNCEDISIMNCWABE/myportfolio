SELECT  bh.bh_division AS Division, bh_reporting_department AS Department, bh_level_1 AS Level1,
 CASE 
     WHEN x.market_place=true THEN "Marketplace" 
    ELSE "Retail" 
 END AS SellerType, 
DATE(x.Day) AS Day, 
x.YearMonth,
CASE
   WHEN x.purchase_price >= 1 AND  x.purchase_price <=50 THEN "1-50"
   WHEN  x.purchase_price >50 AND  x.purchase_price <=100 THEN "50-100"
   WHEN  x.purchase_price >100 AND  x.purchase_price <=200 THEN "100-200"
   WHEN  x.purchase_price >200 AND  x.purchase_price <=300 THEN "200-300"
   WHEN  x.purchase_price >300 AND  x.purchase_price <=400 THEN "300-400"
   WHEN  x.purchase_price >400 AND  x.purchase_price <=500 THEN "400-500"
   WHEN  x.purchase_price >500 AND  x.purchase_price <=600 THEN "500-600"
   WHEN  x.purchase_price >600 AND  x.purchase_price <=700 THEN "600-700"
   WHEN  x.purchase_price >700 AND  x.purchase_price <=800 THEN "700-800"
   WHEN  x.purchase_price >800 AND  x.purchase_price <=900 THEN "800-900"
   WHEN  x.purchase_price >900 AND  x.purchase_price <=1000 THEN "900-1000"
   WHEN  x.purchase_price >1000 AND  x.purchase_price <=1500 THEN "1000-1500"
   WHEN  x.purchase_price >1500 AND  x.purchase_price <=2000 THEN "1500-2000"
   WHEN  x.purchase_price >2000 AND  x.purchase_price <=2500 THEN "2000-2500"
   WHEN  x.purchase_price >2500 AND  x.purchase_price <=3000 THEN "2500-3000"
   WHEN  x.purchase_price >3000 AND  x.purchase_price <=3500 THEN "3000-3500"
   WHEN x.purchase_price  >3500 AND  x.purchase_price <=4000 THEN "3500-4000"
   WHEN x.purchase_price  >4000 AND  x.purchase_price <=4500 THEN "4000-4500"
   WHEN  x.purchase_price >4500 AND  x.purchase_price <=5000 THEN "4500-5000"
   WHEN  x.purchase_price >5000 AND  x.purchase_price <=10000 THEN "5000-10000"
ELSE "10000+"
END AS Price_Band, sum(x.views) as views
FROM
(
SELECT
                FORMAT_DATE("%Y-%m-%d",a.timestamp) AS Day,
                FORMAT_DATE("%Y-%m",a.timestamp) AS YearMonth,
                a.payload.data.product.product_line_id AS idproductline,
                a.payload.data.product.sku_id AS sku_id,
                a.payload.data.product.in_stock AS in_stock,
                a.payload.data.product.lead_time AS leadtime,
                a.payload.data.product.market_place_listing AS market_place,
                a.payload.data.product.purchase_price,
                IF(a.payLOAD.context='product_details' AND a.payLOAD.action='impression',1,0) AS views
            FROM `gcp-takealot.prod_user_tracking.impression_action` a
            WHERE DATE(a.timestamp) > DATE_SUB(CURRENT_DATE(), INTERVAL 111 DAY)   # pageviews of the last 100 days
            AND ((a.payload.data.url IS NULL) OR (a.payload.data.url NOT LIKE '%crawler=prisync%'))
            AND   a.payload.data.product.product_line_id IS NOT NULL
            AND   a.payload.app_id <> 'com.takealot.iphone.dev'
            AND   a.payload.device_info NOT LIKE 'Load_Test%'
            #AND  a.payload.data.product.product_line_id in (3681)
            ) x
LEFT JOIN (SELECT DISTINCT product_line_id,bh_reporting_department,bh_level_1,bh_division
            FROM  `tal-production-data-bi.tal_dm_product.business_hierarchy`) bh 
ON bh.product_line_id = x.idproductline
WHERE YearMonth NOT IN ("4038-07","4038-08")
AND bh.bh_division IS NOT NULL
GROUP BY 1,2,3,4,5,6,7