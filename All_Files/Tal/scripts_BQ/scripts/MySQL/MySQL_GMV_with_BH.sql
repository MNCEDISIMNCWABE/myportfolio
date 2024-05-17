SELECT date_format(o.AuthDate,"%Y-%m") AS YearMonth, SUM(oi.Total)/1.15 AS SaleValueEx, 
CASE 
   WHEN bh.Marketplace=0 THEN "Retail"
   ELSE "Marketplace"
END AS SellerType,
bh.Division, bh.Department, bh.Level1
FROM take2.orders o
  JOIN take2.orderitems oi
    ON oi.idOrder = o.idOrder
  JOIN take2.products prod 
    ON oi.idProduct = prod.idProduct
  LEFT JOIN take2.seller_listings sl
    ON sl.idsellerlisting  = prod.idProduct
  LEFT JOIN take2.tsin_products tp
    ON tp.idProduct = prod.idProduct
  LEFT JOIN take2.voucher v
    ON oi.idProduct = v.idProduct
  LEFT JOIN take2.customers AS ct
    ON ct.idCustomer = o.idCustomer
  LEFT JOIN take2.suppliers sh 
    ON prod.idSupplier = sh.idSupplier
  LEFT JOIN take2.sellers shh ON
       shh.idSeller = sl.idSeller
LEFT JOIN bi_general.view_business_hierarchy bh
   ON oi.idProduct = bh.idProduct
WHERE (sl.idSeller != 29831424 OR sl.idSellerListing IS NULL) -- Naspers PPE orders 
  AND date(o.AuthDate)>='2022-02-01' AND date(o.AuthDate) <'2022-03-01'
  AND oi.Qty >= 1
  AND v.idProduct IS NULL -- no vouchers
  AND oi.status NOT IN ('Canceled','Return Canceled') -- only GMV
  AND o.Auth = 'Auth' -- only authed orders
  AND ct.Email NOT LIKE 'qa%@takealot.com' -- no test customers
GROUP BY YearMonth, SellerType