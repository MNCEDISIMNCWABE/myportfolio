-- BigQuery query, using Data lakehouse tables

WITH orders AS (
    SELECT
        DATETIME(TIMESTAMP_MILLIS(after.AuthDate)) AS AuthDate,
        after.Auth,
        after.idOrder
    FROM
    (
        SELECT *,  row_number()
        OVER(partition by after.idOrder ORDER BY kafka_data.insertTime DESC) as row_number
        FROM `tal-production-data-lakehouse.tal_dl_cdc_mysql_take2.take2_orders`
    )
    WHERE row_number = 1 AND after.idOrder IS NOT NULL
),
orderitems AS (
    SELECT
        after.Status,
        after.idOrder,
        after.idProduct,
        after.Qty
    FROM
    (
        SELECT *,  row_number()
        OVER(partition by after.idOrderItem ORDER BY kafka_data.insertTime DESC) as row_number
        FROM `tal-production-data-lakehouse.tal_dl_cdc_mysql_take2.take2_orderitems`
    )
    WHERE row_number = 1 AND after.idOrderItem IS NOT NULL
),
returns AS (
    SELECT
        after.RRN,
        after.idCustomer,
        after.idProduct,
        after.idOrderItem,
        after.Qty,
        after.DetailsReason,
    FROM
    (
        SELECT *,  row_number()
        OVER(partition by after.idRrnReturn ORDER BY kafka_data.insertTime DESC) as row_number
        FROM `tal-production-data-lakehouse.tal_dl_cdc_mysql_take2.take2_rrn_returns`
    )
    WHERE row_number = 1 AND after.idRrnReturn IS NOT NULL
),

returns_products AS (
    SELECT RRN.idProduct
        , COUNT(RRN.RRN) AS Cnt_Returns
    FROM returns RRN
    WHERE RRN.DetailsReason IN ('The product is defective or damaged', 'The product is not what I ordered or not as described')
    GROUP BY RRN.idProduct
)

SELECT OI.idProduct
    , DATE(O.AuthDate) AS Auth_Date
	, SUM(OI.Qty) AS SalesQty
FROM orders O
    JOIN orderitems OI ON OI.idOrder = O.idOrder
    LEFT JOIN returns_products RRN ON RRN.idProduct = OI.idProduct
WHERE O.Auth = 'Auth'
    AND OI.Status NOT IN ('Canceled')
    AND RRN.Cnt_Returns  >= 1
GROUP BY OI.idProduct, Auth_Date
ORDER BY OI.idProduct, Auth_Date






