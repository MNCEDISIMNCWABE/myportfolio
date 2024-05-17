-- Pull data to see the last sale date of a product from each warehouse
SELECT 
    distinct idTsin, 
    orders.warehouse, 
    max(authdate) as last_sale_date 
FROM take2.orders 
inner join take2.orderitems using (idorder) 
inner join take2.tsin_products on tsin_products.idproduct = orderitems.idproduct 
where idTsin in (76650988)
and DateCanceled is null and DateReturnCanceled is null 
and authdate is not null 
and orders.warehouse is not null
group by idTsin, orders.warehouse