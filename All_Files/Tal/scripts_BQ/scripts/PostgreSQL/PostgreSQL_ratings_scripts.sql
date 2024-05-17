SELECT 
  review.*
FROM review_text
inner join review 
   on review.customer_id = review_text.customer_id and review.tsin_id = review_text.tsin_id
where review_text.tsin_id in (8)
and moderation_state_id = 0