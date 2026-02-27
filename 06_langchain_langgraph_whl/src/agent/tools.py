"""
Customer Support Agent tools - LangChain @tool-decorated functions.
Covers order handling, product info, and refund processing.
"""

from langchain_core.tools import tool


@tool
def get_order_status(order_id: str) -> str:
    """Get the current status of an order by order ID. Use for questions about order status, shipment, or delivery."""
    orders = {
        "ORD-12345678": "Shipped - In transit, estimated delivery Feb 28",
        "ORD-87654321": "Delivered on Feb 20",
        "ORD-11111111": "Processing - Expected to ship within 2 days",
    }
    return orders.get(order_id.upper(), f"Order {order_id} not found. Please verify the order ID.")


@tool
def get_product_info(product_id: str) -> str:
    """Get product details including name, description, and price. Use for product information questions."""
    products = {
        "PROD-0001": "Product: Wireless Headphones Pro | Price: $149.99 | Features: 30hr battery, noise cancellation, Bluetooth 5.3",
        "PROD-0002": "Product: USB-C Hub 7-in-1 | Price: $49.99 | Features: HDMI, USB 3.0, SD card reader",
    }
    return products.get(product_id.upper(), f"Product {product_id} not found in catalog.")


@tool
def get_tracking_info(order_id: str) -> str:
    """Get tracking information for a shipped order. Use when the customer wants to track their package."""
    tracking = {
        "ORD-12345678": "Tracking: 1Z999AA10123456784 | Carrier: UPS | Last update: Departed Memphis, TN",
        "ORD-87654321": "Delivered. Final scan: Feb 20, 2:34 PM at front door.",
    }
    return tracking.get(order_id.upper(), f"No tracking found for {order_id}. Order may not be shipped yet.")


@tool
def check_inventory(product_id: str) -> str:
    """Check if a product is in stock and how many units are available."""
    inventory = {
        "PROD-0001": "In stock: 234 units",
        "PROD-0002": "Low stock: 12 units",
    }
    return inventory.get(product_id.upper(), f"Inventory status unknown for {product_id}.")


@tool
def process_refund(order_id: str, reason: str) -> str:
    """Process a refund request for an order. Use when customer wants to return, refund, or report defective items."""
    return (
        f"Refund request submitted for order {order_id} (reason: {reason}). "
        "Our team will review within 24-48 hours. You will receive an email with the outcome."
    )
