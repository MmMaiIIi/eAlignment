# Evaluation Report

- mode: comparison
- samples: 5
- badcases: 5
- tuned_proxy_score_avg: 0.2500
- base_proxy_score_avg: 0.2500
- delta_proxy_score_avg: 0.0000

## Proxy Metrics Note
Proxy metrics are heuristic directional checks and should not be treated as benchmark truth.

## Category Breakdown
- shipping_logistics: samples=1, tuned_avg=0.2500, base_avg=0.2500, delta_avg=0.0000, badcases=1
- complaint_soothing: samples=1, tuned_avg=0.2500, base_avg=0.2500, delta_avg=0.0000, badcases=1
- returns_refunds: samples=1, tuned_avg=0.2500, base_avg=0.2500, delta_avg=0.0000, badcases=1
- order_modification: samples=1, tuned_avg=0.2500, base_avg=0.2500, delta_avg=0.0000, badcases=1
- product_specs: samples=1, tuned_avg=0.2500, base_avg=0.2500, delta_avg=0.0000, badcases=1

## Top Badcases
### Case 1
- id: eval_0001
- category: shipping_logistics
- reasons: low_proxy_score, low_quality
- prompt: 客户：我的包裹已经 4 天没有物流更新了。
- tuned_response: 很抱歉给您带来等待。请提供订单号，我来为您核实最新物流状态，并告知后续处理方案。

### Case 2
- id: eval_0002
- category: complaint_soothing
- reasons: low_proxy_score, low_quality
- prompt: 客户：你们从来都解决不了任何问题。
- tuned_response: 很抱歉这次体验让您失望了，也理解您的着急。请提供订单号，我会立即帮您核查，并尽快给您明确的处理结果。

### Case 3
- id: eval_0003
- category: returns_refunds
- reasons: low_proxy_score, low_quality
- prompt: 客户：我收到的商品是坏的。
- tuned_response: 很抱歉出现这种情况。请提供订单号，并上传商品破损照片，我来协助您申请退货或换货。

### Case 4
- id: eval_0004
- category: order_modification
- reasons: low_proxy_score, low_quality
- prompt: 客户：付款后我还能修改地址吗？
- tuned_response: 请提供订单号和新的收货地址。如果订单还未发货，我会立即为您提交地址修改申请。

### Case 5
- id: eval_0005
- category: product_specs
- reasons: low_proxy_score, low_quality
- prompt: 客户：这款手机支持双卡吗？
- tuned_response: 是的，这款手机支持双卡。如果您告诉我所在地区，我也可以进一步帮您确认支持的网络频段。
