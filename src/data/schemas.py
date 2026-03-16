from typing import Dict, Iterable, List


DOMAIN_CATEGORIES = [
    "returns_refunds",
    "shipping_logistics",
    "product_specs",
    "order_modification",
    "after_sales",
    "complaint_soothing",
]


def validate_record_fields(record: Dict[str, str], required_fields: Iterable[str]) -> List[str]:
    missing = [field for field in required_fields if field not in record]
    return missing
