SELECT 
   soh.idTsin, 
   DATE(SnapshotDate) AS Date, 
   SUM(SOH_Value) AS SOH_Value 
FROM retail.SOH_AND_BUYABLE_2019 soh
WHERE soh.SnapshotDate IN ('2019-04-30')
   AND soh.SellerType = 'Retail' 
   AND soh.Buyable=1
GROUP BY idTsin, Date
UNION ALL
SELECT 
   so.idTsin,
   DATE(SnapshotDate) AS Date, 
   SUM(SOH_Value) AS SOH_Value 
FROM retail.SOH_AND_BUYABLE_2020 so
WHERE so.SnapshotDate IN ('2020-01-31', '2020-02-28') 
   AND so.SellerType = 'Retail' 
   AND so.Buyable=1
GROUP BY idTsin, Date