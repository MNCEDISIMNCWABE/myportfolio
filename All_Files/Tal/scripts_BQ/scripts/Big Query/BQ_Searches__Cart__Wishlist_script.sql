-- script pulls searches, add to cart, wishlist and pdp data

SELECT sc.productline_id, 
       bh.product_id,
       bh.tsin_id,
       SUM(sc.search_result_cnt) AS searches,
       SUM(pdp_view_cnt) as pdp,
       SUM(add_to_cart_cnt) as add_to_cart,
       SUM(add_to_wish_list_cnt) as wishlist,
       sc.date_day,
       bh.tsin_title,
       bh.bh_division,
       bh.bh_reporting_department,
       bh.bh_level_1,
       bh.bh_level_2,
       bh.bh_level_3,
       bh.brand_name
FROM `gcp-takealot.prod_aggregations.plid_day`  sc
LEFT JOIN `tal-production-data-bi.tal_dm_product.business_hierarchy` bh
          ON sc.productline_id = bh.product_line_id
WHERE date_day >= "2022-04-01" AND date_day < "2022-04-02"   --- One day
GROUP BY sc.productline_id,
       bh.product_id,
       bh.tsin_id,
       sc.date_day,
       bh.bh_division,
       bh.bh_reporting_department,
       bh.bh_level_1,
       bh.bh_level_2,
       bh.bh_level_3,
       bh.brand_name,
       bh.tsin_title;