WITH order_info as(
  SELECT business_user_id, DATE_TRUNC('week',created_at) as week, COUNT(order_uuid) as num_orders
  FROM order_details_view
  WHERE payed = TRUE and status != 'canceled'
  GROUP BY business_user_id,week
  )

SELECT DISTINCT(business_user_id) as user_id
FROM order_info
WHERE num_orders >= 2