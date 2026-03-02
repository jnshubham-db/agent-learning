"""
Tool functions for the Customer Order Support Agent.
Each tool queries the sjdatabricks catalog via SparkSession and returns
a human-readable string summary.
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def get_order_details(order_id: int) -> str:
    """Query sjdatabricks.orders.order_details for a specific order.

    Args:
        order_id: The numeric order ID to look up.

    Returns:
        A formatted string with order details, or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
    )
    rows = df.collect()
    if not rows:
        return f"No order found with order_id={order_id}."
    row = rows[0]
    return (
        f"Order {row['order_id']}: customer={row['customer_name']}, "
        f"product={row['product']}, quantity={row['quantity']}, "
        f"status={row['status']}, order_date={row['order_date']}"
    )


def search_returns(order_id: int) -> str:
    """Query sjdatabricks.orders.returns for returns associated with an order.

    Args:
        order_id: The numeric order ID to search returns for.

    Returns:
        A formatted string listing all returns, or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
    )
    rows = df.collect()
    if not rows:
        return f"No returns found for order_id={order_id}."
    lines = []
    for row in rows:
        lines.append(
            f"Return {row['return_id']}: order_id={row['order_id']}, "
            f"reason={row['reason']}, status={row['status']}, "
            f"return_date={row['return_date']}"
        )
    return "\n".join(lines)


def get_product_info(product_name: str) -> str:
    """Query sjdatabricks.orders.products for a product by name (case-insensitive).

    Args:
        product_name: The product name to search for.

    Returns:
        A formatted string with product details, or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.products WHERE LOWER(name) = LOWER('{product_name}')"
    )
    rows = df.collect()
    if not rows:
        return f"No product found matching '{product_name}'."
    row = rows[0]
    return (
        f"Product {row['product_id']}: name={row['name']}, "
        f"category={row['category']}, price=${row['price']:.2f}, "
        f"stock={row['stock']} units"
    )
