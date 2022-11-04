SELECT business_user_id, SUM(total) as month_total
from order_details_view
WHERE payed = TRUE and status != 'canceled' and created_at >= NOW() - INTERVAL '30 days'
GROUP BY business_user_id
ORDER BY SUM(total) DESC
LIMIT 10