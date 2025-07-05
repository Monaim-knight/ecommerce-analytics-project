-- E-Commerce Analytics SQL Queries
-- Demonstrates advanced SQL skills, data modeling, and business intelligence

-- ===========================================
-- 1. PRODUCT PERFORMANCE ANALYSIS
-- ===========================================

-- Top performing products by revenue
SELECT 
    p.SKU_Code,
    p.Category,
    p.Size,
    p.Color,
    p.Stock,
    p.Final_MRP_Old as Price,
    (p.Stock * p.Final_MRP_Old) as Potential_Revenue,
    RANK() OVER (ORDER BY (p.Stock * p.Final_MRP_Old) DESC) as Revenue_Rank
FROM products p
WHERE p.Stock > 0
ORDER BY Potential_Revenue DESC
LIMIT 20;

-- Category performance analysis
SELECT 
    Category,
    COUNT(*) as Product_Count,
    SUM(Stock) as Total_Stock,
    AVG(Final_MRP_Old) as Avg_Price,
    SUM(Stock * Final_MRP_Old) as Total_Potential_Revenue,
    AVG(Margin_Percentage) as Avg_Margin_Percentage
FROM products
WHERE Stock > 0
GROUP BY Category
ORDER BY Total_Potential_Revenue DESC;

-- Size distribution analysis
SELECT 
    Size,
    COUNT(*) as Product_Count,
    SUM(Stock) as Total_Stock,
    AVG(Final_MRP_Old) as Avg_Price,
    SUM(Stock * Final_MRP_Old) as Total_Revenue
FROM products
WHERE Stock > 0 AND Size != 'FREE'
GROUP BY Size
ORDER BY Total_Revenue DESC;

-- ===========================================
-- 2. INVENTORY OPTIMIZATION QUERIES
-- ===========================================

-- Low stock alert (products with stock < 10)
SELECT 
    SKU_Code,
    Category,
    Size,
    Color,
    Stock,
    Final_MRP_Old as Price,
    CASE 
        WHEN Stock = 0 THEN 'Out of Stock'
        WHEN Stock < 5 THEN 'Critical Low'
        WHEN Stock < 10 THEN 'Low Stock'
        ELSE 'Adequate'
    END as Stock_Status
FROM products
WHERE Stock < 10
ORDER BY Stock ASC;

-- Overstocked products (stock > 100)
SELECT 
    SKU_Code,
    Category,
    Size,
    Color,
    Stock,
    Final_MRP_Old as Price,
    (Stock * Final_MRP_Old) as Inventory_Value
FROM products
WHERE Stock > 100
ORDER BY Stock DESC;

-- Inventory turnover analysis
WITH inventory_metrics AS (
    SELECT 
        Category,
        COUNT(*) as Product_Count,
        SUM(Stock) as Total_Stock,
        AVG(Stock) as Avg_Stock,
        SUM(Stock * Final_MRP_Old) as Total_Inventory_Value
    FROM products
    WHERE Stock > 0
    GROUP BY Category
)
SELECT 
    Category,
    Product_Count,
    Total_Stock,
    ROUND(Avg_Stock, 2) as Avg_Stock,
    ROUND(Total_Inventory_Value, 2) as Total_Inventory_Value,
    ROUND(Total_Inventory_Value / Total_Stock, 2) as Avg_Product_Value
FROM inventory_metrics
ORDER BY Total_Inventory_Value DESC;

-- ===========================================
-- 3. PRICING STRATEGY ANALYSIS
-- ===========================================

-- Price range analysis by category
SELECT 
    Category,
    MIN(Final_MRP_Old) as Min_Price,
    MAX(Final_MRP_Old) as Max_Price,
    AVG(Final_MRP_Old) as Avg_Price,
    STDDEV(Final_MRP_Old) as Price_StdDev,
    COUNT(*) as Product_Count
FROM products
WHERE Final_MRP_Old > 0
GROUP BY Category
ORDER BY Avg_Price DESC;

-- Margin analysis
SELECT 
    Category,
    AVG(Margin_Percentage) as Avg_Margin_Percentage,
    MIN(Margin_Percentage) as Min_Margin_Percentage,
    MAX(Margin_Percentage) as Max_Margin_Percentage,
    COUNT(*) as Product_Count
FROM products
WHERE Margin_Percentage IS NOT NULL
GROUP BY Category
ORDER BY Avg_Margin_Percentage DESC;

-- Price elasticity analysis (products with different pricing across platforms)
SELECT 
    SKU_Code,
    Category,
    Amazon_MRP,
    Flipkart_MRP,
    Myntra_MRP,
    (Amazon_MRP - Flipkart_MRP) as Price_Diff_Amazon_Flipkart,
    CASE 
        WHEN Amazon_MRP > Flipkart_MRP THEN 'Amazon Higher'
        WHEN Flipkart_MRP > Amazon_MRP THEN 'Flipkart Higher'
        ELSE 'Same Price'
    END as Price_Strategy
FROM products
WHERE Amazon_MRP IS NOT NULL AND Flipkart_MRP IS NOT NULL
ORDER BY ABS(Amazon_MRP - Flipkart_MRP) DESC;

-- ===========================================
-- 4. CROSS-PLATFORM PERFORMANCE
-- ===========================================

-- Platform pricing comparison
SELECT 
    Category,
    AVG(Amazon_MRP) as Avg_Amazon_Price,
    AVG(Flipkart_MRP) as Avg_Flipkart_Price,
    AVG(Myntra_MRP) as Avg_Myntra_Price,
    AVG(Paytm_MRP) as Avg_Paytm_Price,
    COUNT(*) as Product_Count
FROM products
WHERE Amazon_MRP IS NOT NULL OR Flipkart_MRP IS NOT NULL
GROUP BY Category
ORDER BY Product_Count DESC;

-- Platform-specific margin analysis
SELECT 
    'Amazon' as Platform,
    AVG(Amazon_MRP - TP) as Avg_Margin,
    COUNT(*) as Product_Count
FROM products
WHERE Amazon_MRP IS NOT NULL AND TP IS NOT NULL
UNION ALL
SELECT 
    'Flipkart' as Platform,
    AVG(Flipkart_MRP - TP) as Avg_Margin,
    COUNT(*) as Product_Count
FROM products
WHERE Flipkart_MRP IS NOT NULL AND TP IS NOT NULL
UNION ALL
SELECT 
    'Myntra' as Platform,
    AVG(Myntra_MRP - TP) as Avg_Margin,
    COUNT(*) as Product_Count
FROM products
WHERE Myntra_MRP IS NOT NULL AND TP IS NOT NULL;

-- ===========================================
-- 5. FINANCIAL ANALYSIS
-- ===========================================

-- Revenue potential by category
SELECT 
    Category,
    SUM(Stock * Final_MRP_Old) as Total_Revenue_Potential,
    SUM(Stock * TP) as Total_Cost,
    SUM(Stock * (Final_MRP_Old - TP)) as Total_Profit_Potential,
    ROUND((SUM(Stock * (Final_MRP_Old - TP)) / SUM(Stock * Final_MRP_Old)) * 100, 2) as Profit_Margin_Percentage
FROM products
WHERE Stock > 0 AND Final_MRP_Old > 0 AND TP > 0
GROUP BY Category
ORDER BY Total_Profit_Potential DESC;

-- Expense analysis
SELECT 
    Particular,
    SUM(Amount) as Total_Expense,
    COUNT(*) as Transaction_Count,
    AVG(Amount) as Avg_Expense
FROM expenses
WHERE Amount > 0
GROUP BY Particular
ORDER BY Total_Expense DESC;

-- Profit and Loss summary
WITH revenue_summary AS (
    SELECT SUM(Stock * Final_MRP_Old) as Total_Revenue_Potential
    FROM products
    WHERE Stock > 0
),
expense_summary AS (
    SELECT SUM(Amount) as Total_Expenses
    FROM expenses
    WHERE Amount > 0
)
SELECT 
    r.Total_Revenue_Potential,
    e.Total_Expenses,
    (r.Total_Revenue_Potential - e.Total_Expenses) as Net_Profit,
    ROUND(((r.Total_Revenue_Potential - e.Total_Expenses) / r.Total_Revenue_Potential) * 100, 2) as Profit_Margin_Percentage
FROM revenue_summary r, expense_summary e;

-- ===========================================
-- 6. ADVANCED ANALYTICS QUERIES
-- ===========================================

-- Product portfolio analysis (ABC analysis)
WITH product_analysis AS (
    SELECT 
        SKU_Code,
        Category,
        (Stock * Final_MRP_Old) as Revenue_Potential,
        ROUND((Stock * Final_MRP_Old) / SUM(Stock * Final_MRP_Old) OVER() * 100, 2) as Revenue_Percentage
    FROM products
    WHERE Stock > 0
),
ranked_products AS (
    SELECT 
        *,
        SUM(Revenue_Percentage) OVER (ORDER BY Revenue_Percentage DESC) as Cumulative_Percentage
    FROM product_analysis
)
SELECT 
    SKU_Code,
    Category,
    Revenue_Potential,
    Revenue_Percentage,
    CASE 
        WHEN Cumulative_Percentage <= 80 THEN 'A'
        WHEN Cumulative_Percentage <= 95 THEN 'B'
        ELSE 'C'
    END as Product_Class
FROM ranked_products
ORDER BY Revenue_Percentage DESC;

-- Seasonal analysis (if date data available)
SELECT 
    EXTRACT(MONTH FROM Date) as Month,
    COUNT(*) as Transaction_Count,
    SUM(Amount) as Total_Amount,
    AVG(Amount) as Avg_Amount
FROM transactions
WHERE Date IS NOT NULL
GROUP BY EXTRACT(MONTH FROM Date)
ORDER BY Month;

-- Customer segmentation (if customer data available)
SELECT 
    Customer_Segment,
    COUNT(*) as Customer_Count,
    AVG(Order_Value) as Avg_Order_Value,
    SUM(Total_Revenue) as Total_Revenue,
    COUNT(*) / SUM(COUNT(*)) OVER() * 100 as Customer_Percentage
FROM customers
GROUP BY Customer_Segment
ORDER BY Total_Revenue DESC;

-- ===========================================
-- 7. PREDICTIVE ANALYTICS QUERIES
-- ===========================================

-- Demand forecasting (using historical data)
WITH demand_trends AS (
    SELECT 
        Category,
        AVG(Stock) as Avg_Stock,
        STDDEV(Stock) as Stock_StdDev,
        COUNT(*) as Product_Count
    FROM products
    WHERE Stock > 0
    GROUP BY Category
)
SELECT 
    Category,
    Avg_Stock,
    Stock_StdDev,
    (Avg_Stock + (2 * Stock_StdDev)) as Forecasted_Demand,
    Product_Count
FROM demand_trends
ORDER BY Forecasted_Demand DESC;

-- Price optimization recommendations
SELECT 
    SKU_Code,
    Category,
    Final_MRP_Old as Current_Price,
    TP as Cost_Price,
    Margin_Percentage,
    CASE 
        WHEN Margin_Percentage < 20 THEN 'Increase Price'
        WHEN Margin_Percentage > 50 THEN 'Consider Discount'
        ELSE 'Optimal Price'
    END as Price_Recommendation,
    CASE 
        WHEN Margin_Percentage < 20 THEN Final_MRP_Old * 1.1
        WHEN Margin_Percentage > 50 THEN Final_MRP_Old * 0.9
        ELSE Final_MRP_Old
    END as Recommended_Price
FROM products
WHERE Margin_Percentage IS NOT NULL
ORDER BY Margin_Percentage;

-- ===========================================
-- 8. REAL-TIME MONITORING QUERIES
-- ===========================================

-- Real-time inventory status
SELECT 
    COUNT(*) as Total_Products,
    SUM(CASE WHEN Stock = 0 THEN 1 ELSE 0 END) as Out_of_Stock,
    SUM(CASE WHEN Stock < 10 THEN 1 ELSE 0 END) as Low_Stock,
    SUM(CASE WHEN Stock > 100 THEN 1 ELSE 0 END) as Overstocked,
    ROUND((SUM(CASE WHEN Stock = 0 THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) as Out_of_Stock_Percentage
FROM products;

-- Revenue alerts
SELECT 
    'High Value Products' as Alert_Type,
    COUNT(*) as Product_Count,
    SUM(Stock * Final_MRP_Old) as Total_Value
FROM products
WHERE (Stock * Final_MRP_Old) > 10000
UNION ALL
SELECT 
    'Low Margin Products' as Alert_Type,
    COUNT(*) as Product_Count,
    SUM(Stock * Final_MRP_Old) as Total_Value
FROM products
WHERE Margin_Percentage < 10 AND Margin_Percentage IS NOT NULL;

-- ===========================================
-- 9. PERFORMANCE OPTIMIZATION QUERIES
-- ===========================================

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_products_category ON products(Category);
CREATE INDEX IF NOT EXISTS idx_products_stock ON products(Stock);
CREATE INDEX IF NOT EXISTS idx_products_price ON products(Final_MRP_Old);
CREATE INDEX IF NOT EXISTS idx_products_sku ON products(SKU_Code);

-- Partitioned view for large datasets
CREATE VIEW products_partitioned AS
SELECT * FROM products
WHERE Stock > 0
PARTITION BY Category;

-- Materialized view for frequently accessed data
CREATE MATERIALIZED VIEW category_summary AS
SELECT 
    Category,
    COUNT(*) as Product_Count,
    SUM(Stock) as Total_Stock,
    AVG(Final_MRP_Old) as Avg_Price,
    SUM(Stock * Final_MRP_Old) as Total_Revenue
FROM products
WHERE Stock > 0
GROUP BY Category;

-- ===========================================
-- 10. DATA QUALITY QUERIES
-- ===========================================

-- Data quality checks
SELECT 
    'Missing SKU Codes' as Quality_Issue,
    COUNT(*) as Issue_Count
FROM products
WHERE SKU_Code IS NULL OR SKU_Code = ''
UNION ALL
SELECT 
    'Negative Stock' as Quality_Issue,
    COUNT(*) as Issue_Count
FROM products
WHERE Stock < 0
UNION ALL
SELECT 
    'Invalid Prices' as Quality_Issue,
    COUNT(*) as Issue_Count
FROM products
WHERE Final_MRP_Old <= 0
UNION ALL
SELECT 
    'Missing Categories' as Quality_Issue,
    COUNT(*) as Issue_Count
FROM products
WHERE Category IS NULL OR Category = '';

-- Duplicate detection
SELECT 
    SKU_Code,
    COUNT(*) as Duplicate_Count
FROM products
GROUP BY SKU_Code
HAVING COUNT(*) > 1
ORDER BY Duplicate_Count DESC; 