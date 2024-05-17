# ------------------------------------------- Returns----------------------------------------
SELECT 
#SUM(oi.order_item_quantity) AS Qty_returned,
SUM(oi.order_item_sales) AS GMV_Ex,
FORMAT_DATE("%Y%m",order_placed_timestamp) AS Auth_Date,
CASE 
   WHEN oi.is_marketplace_item = false THEN "Retail"
   ELSE "Marketplace"
END AS SellerType
FROM `tal-production-data-bi.tal_dm_orders.orders_2020_2030` od,
UNNEST(od.order_items) oi
WHERE   
        DATE(order_placed_timestamp) >= "2021-01-01" AND DATE(order_placed_timestamp) < "2022-02-01"
        #AND oi.is_authed_not_return_cancel IS FALSE
        #AND oi.order_item_status IN ('Return Canceled','Canceled')
        #AND oi.order_item_status LIKE 'Shipped'
        AND order_authorisation_status IN ('Auth')
        AND is_returned = true
        #AND is_cancelled = true
        AND is_voucher = false
        AND is_test_customer = false
        #AND is_marketplace_item = false
GROUP BY SellerType, Auth_Date
ORDER BY Auth_Date ASC

# ------------------------------------------- Cancellations ----------------------------------------
SELECT 
#SUM(oi.order_item_quantity) AS Qty_returned,
SUM(oi.order_item_sales) AS GMV_Ex,
FORMAT_DATE("%Y%m",order_placed_timestamp) AS Auth_Date,
CASE 
   WHEN oi.is_marketplace_item = false THEN "Retail"
   ELSE "Marketplace"
END AS SellerType
FROM `tal-production-data-bi.tal_dm_orders.orders_2020_2030` od,
UNNEST(od.order_items) oi
WHERE  
        DATE(order_placed_timestamp) >= "2021-01-01" AND DATE(order_placed_timestamp) < "2022-02-01"
        AND oi.is_authed_not_return_cancel IS FALSE
        AND oi.order_item_status IN ('Return Canceled','Canceled')
        #AND oi.order_item_status LIKE 'Shipped'
        #AND order_authorisation_status IN ('Auth')
        #AND is_returned = true
        AND is_cancelled = true
        AND is_voucher = false
        AND is_test_customer = false
        #AND is_marketplace_item = false
GROUP BY SellerType , Auth_Date
ORDER BY Auth_Date ASC