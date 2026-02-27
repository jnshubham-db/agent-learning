"""
Mock tools for the Customer Support Agent.
These simulate backend API calls for order status, product info, and refund processing.
"""


def get_order_status(order_id: str) -> str:
    """
    Get the status of an order by its ID.

    Args:
        order_id: The order identifier (e.g., ORD-12345678).

    Returns:
        Order status, tracking info, and estimated delivery when available.
    """
    order_db = {
        "ORD-12345678": {
            "status": "Shipped",
            "tracking": "1Z999AA10123456784",
            "eta": "Feb 28, 2025",
        },
        "ORD-87654321": {"status": "Delivered", "delivered_date": "Feb 25, 2025"},
        "ORD-11111111": {"status": "Processing", "estimated_ship": "Mar 1, 2025"},
    }
    order_id_upper = order_id.upper() if isinstance(order_id, str) else order_id
    if order_id_upper in order_db:
        info = order_db[order_id_upper]
        parts = [f"Order {order_id_upper}: {info.get('status', 'Unknown')}"]
        for k, v in info.items():
            if k != "status":
                parts.append(f"{k}: {v}")
        return ". ".join(parts)
    return f"Order {order_id} not found."


def get_product_info(product_id: str) -> str:
    """
    Get product information by product ID.

    Args:
        product_id: The product identifier (e.g., PROD-0001).

    Returns:
        Product name, price, stock status, and rating.
    """
    product_db = {
        "PROD-0001": {
            "name": "Wireless Headphones",
            "price": "$99.99",
            "in_stock": True,
            "rating": 4.5,
        },
        "PROD-0002": {"name": "USB-C Hub", "price": "$49.99", "in_stock": True, "rating": 4.2},
        "PROD-0003": {
            "name": "Mechanical Keyboard",
            "price": "$129.99",
            "in_stock": False,
            "rating": 4.8,
        },
    }
    product_id_upper = product_id.upper() if isinstance(product_id, str) else product_id
    if product_id_upper in product_db:
        p = product_db[product_id_upper]
        return (
            f"{p['name']} ({product_id_upper}): {p['price']}, "
            f"In stock: {p['in_stock']}, Rating: {p['rating']}/5"
        )
    return f"Product {product_id} not found."


def process_refund(order_id: str, reason: str) -> str:
    """
    Process a refund request for an order.

    Args:
        order_id: The order identifier to refund.
        reason: Reason for the refund request.

    Returns:
        Confirmation with refund ID and pending status.
    """
    ref_id = f"REF-{str(order_id)[-6:]}" if len(str(order_id)) >= 6 else f"REF-{order_id}"
    return (
        f"Refund request submitted for order {order_id}. "
        f"Reason: {reason}. Refund ID: {ref_id}. Status: Pending approval."
    )
