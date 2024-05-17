-- script pulls monthly gmv, qty sold, no. of orders, no. of customers from 2016

SELECT 
   x.Division, x.Department, x.Level1, x.SellerType, x.YearMonth, 
   COUNT(DISTINCT x.customer_id) AS Nr_Customers, 
   SUM(x.Num_Orders) AS Nr_Orders,
   SUM(x.Qty) AS Qty_sold, 
   SUM(x.GMV) AS GMV_Ex
FROM (
     SELECT 
        bh.bh_division AS Division, bh.bh_reporting_department AS Department, bh.bh_level_1 AS Level1,
        CASE 
           WHEN bh.is_marketplace=true THEN "Marketplace"
          ELSE "Retail"
        END AS SellerType,
        customer_id, FORMAT_DATE("%Y-%m",order_placed_timestamp) AS YearMonth, 
        COUNT(DISTINCT o.order_id) AS Num_Orders,
        SUM(oi.order_item_quantity) AS Qty,
	    SUM(order_item_sales) AS GMV
    FROM `tal-production-data-bi.tal_dm_order.orders_2010_2020` o,
    UNNEST(o.order_items) oi
    INNER JOIN `tal-production-data-bi.tal_dm_product.business_hierarchy` bh
               ON bh.product_id = oi.product_id
WHERE   
        DATE(order_placed_timestamp) >= "2016-01-01" AND DATE(order_placed_timestamp) < "2020-01-01" 
        AND oi.is_authed_not_return_cancel IS TRUE 
        AND oi.order_item_status NOT IN ('Return Canceled','Canceled')
        AND oi.order_item_status LIKE 'Shipped'
        AND order_authorisation_status IN ('Auth')
        AND is_returned = false
        AND is_cancelled = false
        AND is_voucher = false
        AND is_test_customer = false
GROUP BY o.customer_id, YearMonth, bh.bh_division, bh.bh_reporting_department, bh.bh_level_1,customer_id,SellerType

UNION ALL

SELECT 
        bh.bh_division AS Division, bh.bh_reporting_department AS Department, bh.bh_level_1 AS Level1,
        CASE 
           WHEN bh.is_marketplace=true THEN "Marketplace"
          ELSE "Retail"
        END AS SellerType,
        customer_id, FORMAT_DATE("%Y-%m",order_placed_timestamp) AS YearMonth, 
        COUNT(DISTINCT o.order_id) AS Num_Orders,
        SUM(oi.order_item_quantity) AS Qty,
        SUM(order_item_sales) AS GMV
    FROM `tal-production-data-bi.tal_dm_order.orders_2020_2030` o,
    UNNEST(o.order_items) oi
    INNER JOIN `tal-production-data-bi.tal_dm_product.business_hierarchy` bh
               ON bh.product_id = oi.product_id
WHERE   
        DATE(order_placed_timestamp) >= "2020-01-01" AND DATE(order_placed_timestamp) < "2022-04-01" 
        AND oi.is_authed_not_return_cancel IS TRUE 
        AND oi.order_item_status NOT IN ('Return Canceled','Canceled')
        AND oi.order_item_status LIKE 'Shipped'
        AND order_authorisation_status IN ('Auth')
        AND is_returned = false
        AND is_cancelled = false
        AND is_voucher = false
        AND is_test_customer = false
GROUP BY o.customer_id, YearMonth, bh.bh_division, bh.bh_reporting_department, bh.bh_level_1,customer_id,SellerType) x  
GROUP BY  YearMonth, x.Division, x.Department, x.Level1,SellerType
