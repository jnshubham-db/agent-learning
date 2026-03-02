"""
Tool functions for the Customer Support Agent.
Each tool queries Spark tables in the sjdatabricks catalog to retrieve
order details, return information, or product data.
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()


def get_order_details(order_id: int) -> str:
    """
    Look up order details from sjdatabricks.orders.order_details.

    Args:
        order_id: The numeric order identifier.

    Returns:
        Formatted string with order information or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.order_details WHERE order_id = {order_id}"
    )
    rows = df.collect()
    if not rows:
        return f"No order found with ID {order_id}."
    row = rows[0]
    return (
        f"Order {row['order_id']}: "
        f"Customer: {row['customer_name']}, "
        f"Product: {row['product']}, "
        f"Quantity: {row['quantity']}, "
        f"Status: {row['status']}, "
        f"Order Date: {row['order_date']}"
    )


def search_returns(order_id: int) -> str:
    """
    Search for return records associated with an order.

    Args:
        order_id: The numeric order identifier to look up returns for.

    Returns:
        Formatted string with return details or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.returns WHERE order_id = {order_id}"
    )
    rows = df.collect()
    if not rows:
        return f"No returns found for order {order_id}."
    results = []
    for row in rows:
        results.append(
            f"Return {row['return_id']}: "
            f"Order {row['order_id']}, "
            f"Reason: {row['reason']}, "
            f"Status: {row['status']}, "
            f"Return Date: {row['return_date']}"
        )
    return "\n".join(results)


def get_product_info(product_name: str) -> str:
    """
    Look up product information by name (case-insensitive partial match).

    Args:
        product_name: The product name or partial name to search for.

    Returns:
        Formatted string with product details or a not-found message.
    """
    df = spark.sql(
        f"SELECT * FROM sjdatabricks.orders.products "
        f"WHERE LOWER(name) LIKE LOWER('%{product_name}%')"
    )
    rows = df.collect()
    if not rows:
        return f"No products found matching '{product_name}'."
    results = []
    for row in rows:
        results.append(
            f"Product {row['product_id']}: "
            f"{row['name']}, "
            f"Category: {row['category']}, "
            f"Price: ${row['price']}, "
            f"Stock: {row['stock']}"
        )
    return "\n".join(results)
