def aggregate_sales(records):
    total_revenue = 0
    product_revenue = {}
    region_revenue = {}
    
    for product, region, revenue in records:
        total_revenue += revenue
        if product in product_revenue:
            product_revenue[product] += revenue
        else:
            product_revenue[product] = revenue
        if region in region_revenue:
            region_revenue[region] += revenue
        else:
            region_revenue[region] = revenue
    
    top_product = min(product_revenue, key=lambda x: (-product_revenue[x], x))
    top_region = min(region_revenue, key=lambda x: (-region_revenue[x], x))
    
    return {
        'total_revenue': total_revenue,
        'top_product': top_product,
        'top_region': top_region
    }
