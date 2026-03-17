# Evaluation Report

- mode: comparison
- samples: 5
- badcases: 1
- tuned_proxy_score_avg: 0.9000
- base_proxy_score_avg: 0.3500
- delta_proxy_score_avg: 0.5500

## Proxy Metrics Note
Proxy metrics are heuristic directional checks and should not be treated as benchmark truth.

## Category Breakdown
- shipping_logistics: samples=1, tuned_avg=1.0000, base_avg=0.2500, delta_avg=0.7500, badcases=0
- complaint_soothing: samples=1, tuned_avg=1.0000, base_avg=0.5000, delta_avg=0.5000, badcases=0
- returns_refunds: samples=1, tuned_avg=1.0000, base_avg=0.2500, delta_avg=0.7500, badcases=0
- order_modification: samples=1, tuned_avg=1.0000, base_avg=0.2500, delta_avg=0.7500, badcases=0
- product_specs: samples=1, tuned_avg=0.5000, base_avg=0.5000, delta_avg=0.0000, badcases=1

## Top Badcases
### Case 1
- id: eval_0005
- category: product_specs
- reasons: low_proxy_score
- prompt: Customer: Is this phone dual SIM?
- tuned_response: Yes, this model supports dual SIM. If you share your region, I can also confirm network band compatibility.
