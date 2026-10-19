from collections import defaultdict

def aggregate_sales(records):
    total_revenue_dict = defaultdict(float)
    product_revenue_dict = defaultdict(float)
    region_revenue_dict = defaultdict(float)

    for product, region, revenue in records:
        total_revenue_dict[product] += revenue
        product_revenue_dict[product] += revenue
        region_revenue_dict[region] += revenue

    total_revenue = sum(total_revenue_dict.values())
    top_product = max(total_revenue_dict, key=total_revenue_dict.get)
    top_region = max(region_revenue_dict, key=region_revenue_dict.get)

    return {
        "total_revenue": total_revenue,
        "top_product": top_product,
        "top_region": top_region
    }
