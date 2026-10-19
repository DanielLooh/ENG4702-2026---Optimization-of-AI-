def aggregate_sales(records):
    total_revenue = sum(record[2] for record in records)
    top_product = max(records, key=lambda x: x[2])[0]
    top_region = max(records, key=lambda x: x[2])[1]
    return {"total_revenue": total_revenue, "top_product": top_product, "top_region": top_region}
