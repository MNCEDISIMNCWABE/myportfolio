-- This script pulls buybox information ONLY where Retail in losing the buybox in the last 2 hours
-- so any filters applied are in accordance to that

SELECT
      DISTINCT bb_winner.payload.data.product.product_line_id as PLID,
      CASE 
         WHEN bb_winner.payload.data.product.market_place_listing = true THEN "Marketplace"
         ELSE "Retail"
      END AS buybox_winner,
      bb_winner.payload.data.product.purchase_price as buybox_winner_price,
      p_rest.purchase_price AS Retail_Price,
      #bb_winner.payload.data.product.sku_id,
      #COUNT(distinct bb_winner.payload.data.product.market_place_listing) AS freq,
      DATE(FORMAT_DATE("%Y-%m-%d",DATETIME(bb_winner.timestamp,'Africa/Johannesburg'))) AS Date,
      CASE 
         WHEN bb_winner.payload.data.product.in_stock=true THEN "Instock"
         ELSE "LeadTime"
      END AS Marketplace_Instock,
      CASE 
          WHEN p_rest.in_stock=true THEN "Instock"
          ELSE "LeadTime"
      END AS Retail_Instock,
      bh.bh_division AS Division, bh.bh_reporting_department AS Department, bh.bh_level_1 AS Level1,bh.tsin_title , bh.brand_name AS BrandName
  FROM  `gcp-takealot.prod_user_tracking.impression_action` AS bb_winner,
        `gcp-takealot.prod_user_tracking.impression_action` AS bb_rest,
        UNNEST(bb_rest.payload.data.products) p_rest
        LEFT JOIN `tal-production-data-bi.tal_dm_product.business_hierarchy` bh
             ON bb_winner.payload.data.product.product_line_id = bh.product_line_id
  WHERE 
     p_rest.product_line_id = bb_winner.payload.data.product.product_line_id       -- join buybox winners and other offers 
     AND (bb_winner.payload.event_timestamp) BETWEEN TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -2 HOUR) AND CURRENT_TIMESTAMP() -- last two hours data 
     AND (bb_winner.timestamp) BETWEEN TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -2 HOUR) AND CURRENT_TIMESTAMP() 
     AND (bb_rest.payload.event_timestamp) BETWEEN TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -2 HOUR) AND CURRENT_TIMESTAMP()  
     AND (bb_rest.timestamp) BETWEEN TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -2 HOUR) AND CURRENT_TIMESTAMP() 
     AND bb_winner.payload.context = "product_details"                       -- 'product_details' contexts contain the buybox winner
     AND bb_rest.payload.context = "product_details.offers.new"              -- 'product_details.offers.new' contexts contain other offer
     AND bb_winner.payload.data.product.market_place_listing IS NOT NULL
     AND bb_winner.payload.data.product.market_place_listing IN (true)
     AND bb_winner.payload.data.product.market_place_listing <> p_rest.market_place_listing   -- exclude where MP sellers are competiting alone
   #AND bb_winner.payload.data.product.product_line_id IN (20588517,72175717,97965,4731,130633,28019783,50147364) 
GROUP BY 
       buybox_winner,
       Date,
       bb_winner.payload.data.product.product_line_id,
       Retail_Price,
       bb_winner.payload.data.product.purchase_price,
       bb_winner.payload.data.product.in_stock,
       p_rest.in_stock, bh.bh_division, bh.bh_reporting_department, bh.bh_level_1, bh.brand_name, bh.tsin_title
