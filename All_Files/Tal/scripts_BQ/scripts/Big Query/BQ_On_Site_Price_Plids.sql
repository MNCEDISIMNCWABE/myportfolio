-- Pull current on-site price for PLID
SELECT
       DISTINCT(bb_winner.payload.data.product.product_line_id) AS idProductLine,
       bb_winner.payload.data.product.purchase_price AS on_site_price,
       MAX(DATE(bb_winner.timestamp,'Africa/Johannesburg')) AS time_stamp
  FROM `gcp-takealot.prod_user_tracking.impression_action` AS bb_winner
  WHERE 
   -- 'product_details' contexts contain the buybox winner
      DATE(bb_winner.payload.event_timestamp) >= CURRENT_DATE()
      AND DATE(bb_winner.timestamp) >= CURRENT_DATE()
      AND bb_winner.payload.context = "product_details"
      AND bb_winner.payload.data.product.market_place_listing IS NOT NULL
    #AND bb_winner.payload.data.product.product_line_id IN ()
   GROUP BY bb_winner.payload.data.product.product_line_id, on_site_price
 