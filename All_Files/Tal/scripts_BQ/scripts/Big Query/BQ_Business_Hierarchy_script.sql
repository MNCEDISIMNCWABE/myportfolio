SELECT 
    bh.product_line_id, 
    bh.tsin_id,bh.bh_division AS Division, 
    bh.bh_reporting_department AS Department,
    bh.bh_level_1 AS Level1
FROM `tal-production-data-bi.tal_dm_product.business_hierarchy` bh