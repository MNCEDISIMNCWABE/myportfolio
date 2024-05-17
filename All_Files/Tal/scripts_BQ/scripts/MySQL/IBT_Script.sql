SELECT InstructionTypeName, Description,si.idProduct, QtyRequested, QtyAvailable, QtyDamaged,
QtyPutaway, si.DateCreated, i.ETAMin, i.ETAMax,idWarehouseIncoming, idWarehouseOutgoing
FROM take2.wms2_stock_items si
inner join take2.wms2_instructions i
on si.idInstruction=i.idInstruction
inner join take2.wms2_instruction_types it
on i.idInstructionType=it.idInstructionType
where si.DateCreated >= '2021-12-29' AND InstructionTypeName LIKE '%IBT%'
