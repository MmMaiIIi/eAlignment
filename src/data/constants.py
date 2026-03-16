CANONICAL_CATEGORIES = [
    "returns_refunds",
    "shipping_logistics",
    "product_specs",
    "order_modification",
    "after_sales",
    "complaint_soothing",
]

REQUIRED_SFT_FIELDS = ["id", "category", "system", "instruction", "input", "output"]
REQUIRED_DPO_FIELDS = ["id", "category", "prompt", "chosen", "rejected"]

CATEGORY_ALIASES = {
    "returns_refunds": "returns_refunds",
    "return_refund": "returns_refunds",
    "return": "returns_refunds",
    "refund": "returns_refunds",
    "refunds": "returns_refunds",
    "returns": "returns_refunds",
    "shipping_logistics": "shipping_logistics",
    "shipping": "shipping_logistics",
    "logistics": "shipping_logistics",
    "delivery": "shipping_logistics",
    "delivery_status": "shipping_logistics",
    "product_specs": "product_specs",
    "product_spec": "product_specs",
    "product": "product_specs",
    "specs": "product_specs",
    "specification": "product_specs",
    "order_modification": "order_modification",
    "order_change": "order_modification",
    "change_order": "order_modification",
    "modify_order": "order_modification",
    "after_sales": "after_sales",
    "aftersales": "after_sales",
    "warranty": "after_sales",
    "post_sale": "after_sales",
    "complaint_soothing": "complaint_soothing",
    "complaint": "complaint_soothing",
    "soothing": "complaint_soothing",
    "deescalation": "complaint_soothing",
}
