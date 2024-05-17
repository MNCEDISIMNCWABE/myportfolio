-- Current SOH data by storage area and stock health
SELECT TSIN, 
      DC, Storage_Type AS StorageArea, 
      SUM(CurrentQty) AS Current_SOH, 
      SellerType,
	  Division, Department, Level1, 
      ItemDescription,
      Stock_Health, 
      Orig_UnitCube 
FROM supply_chain.SOH_DC_Locked 
WHERE CPY like "TAL"
AND SellerType Like "Retail"  -- Only retail
GROUP BY TSIN, DC, StorageArea, SellerType;