# SQL: Basic → Advanced — 130+ Queries with Twin Practice
A self-study reference. Each item has a **worked query** and a **Twin** (a similar problem + its solution) so you can verify the pattern and move on. Cover the Twin's solution, solve it, then check.

**Dialect:** ANSI/standard SQL. Hive/SparkSQL differences are flagged with `-- Hive/Spark:` notes where they matter (you list HiveQL, so expect these).

---

## Reference schema (all queries use these tables)
```
employees   (emp_id, emp_name, dept_id, manager_id, salary, hire_date, country)
departments (dept_id, dept_name, location, min_salary)
customers   (customer_id, customer_name, signup_date, country)
products    (product_id, product_name, category, price)
orders      (order_id, customer_id, product_id, order_date, quantity, amount, status)
sessions    (session_id, user_id, login_time, logout_time, device)
video_events(event_id, user_id, video_id, creator_id, event_type, event_time)
creators    (creator_id, creator_name, region, join_date)
```

---

## Sample data — EVERY query in this doc runs against exactly these rows
This is the trick to *learning* the queries: don't read a query in the abstract. Run it in your head against this fixed dataset, predict the output, then check. The rows are deliberately small and chosen so the hard queries return interesting answers (ties, category switches, rank changes, a 3rd order, etc.). The full input→output traces for the boss tier are in the appendix at the very bottom.

**departments**
| dept_id | dept_name | location | min_salary |
|--|--|--|--|
| 1 | Engineering | Seattle | 80000 |
| 2 | Sales | NYC | 50000 |
| 3 | Data | Seattle | 90000 |
| 4 | Support | Austin | 40000 |

**employees** (`manager_id` is self-referencing → employees table)
| emp_id | emp_name | dept_id | manager_id | salary | hire_date | country |
|--|--|--|--|--|--|--|
| 1 | Alice | 3 | NULL | 150000 | 2019-03-01 | US |
| 2 | Bob | 3 | 1 | 120000 | 2021-06-15 | US |
| 3 | Carol | 1 | 1 | 130000 | 2020-01-10 | US |
| 4 | Dan | 1 | 3 | 95000 | 2023-09-01 | IN |
| 5 | Eve | 2 | 1 | 70000 | 2022-04-20 | US |
| 6 | Frank | 2 | 5 | 60000 | 2024-02-01 | IN |

**customers**
| customer_id | customer_name | signup_date | country |
|--|--|--|--|
| 101 | Acme | 2025-11-01 | US |
| 102 | Beta | 2025-12-15 | US |
| 103 | Gamma | 2026-01-05 | IN |
| 104 | Delta | 2026-02-20 | UK |

**products**
| product_id | product_name | category | price |
|--|--|--|--|
| 1 | Widget Pro | electronics | 200 |
| 2 | Gizmo | electronics | 50 |
| 3 | Notebook | books | 15 |
| 4 | Desk Lamp | home | 40 |

**orders** (customer 101 has 3 orders spanning electronics → books → home — used by the "3rd order" and "category switch" problems)
| order_id | customer_id | product_id | order_date | quantity | amount | status |
|--|--|--|--|--|--|--|
| 1 | 101 | 1 | 2026-01-10 | 2 | 400 | shipped |
| 2 | 101 | 3 | 2026-02-05 | 5 | 75 | shipped |
| 3 | 102 | 1 | 2026-01-20 | 1 | 200 | delivered |
| 4 | 103 | 2 | 2026-02-15 | 3 | 150 | cancelled |
| 5 | 101 | 4 | 2026-03-01 | 1 | 40 | shipped |
| 6 | 102 | 2 | 2026-03-12 | 5 | 250 | shipped |

**creators**
| creator_id | creator_name | region | join_date |
|--|--|--|--|
| 1 | NovaK | US | 2025-09-01 |
| 2 | RiyaG | IN | 2025-10-15 |
| 3 | LeoM | US | 2026-01-01 |

**video_events** (`event_type` ∈ view/like/purchase)
| event_id | user_id | video_id | creator_id | event_type | event_time |
|--|--|--|--|--|--|
| 1 | 900 | 11 | 1 | view | 2026-01-03 08:00 |
| 2 | 901 | 11 | 1 | like | 2026-01-03 08:05 |
| 3 | 900 | 12 | 1 | view | 2026-02-04 09:00 |
| 4 | 902 | 21 | 2 | view | 2026-01-10 10:00 |
| 5 | 902 | 21 | 2 | like | 2026-01-10 10:01 |
| 6 | 900 | 31 | 3 | view | 2026-02-11 12:00 |
| 7 | 900 | 31 | 3 | purchase | 2026-02-11 12:30 |
| 8 | 901 | 12 | 1 | view | 2026-02-20 14:00 |

**sessions**
| session_id | user_id | login_time | logout_time | device |
|--|--|--|--|--|
| 1 | 900 | 2026-06-01 08:00 | 2026-06-01 08:45 | ios |
| 2 | 900 | 2026-06-01 20:00 | 2026-06-01 20:30 | web |
| 3 | 901 | 2026-06-01 08:30 | 2026-06-01 09:15 | android |
| 4 | 902 | 2026-06-02 11:00 | 2026-06-02 11:20 | ios |

> Some advanced problems reference extra tables not shown above (`trades`, `quotes`, `bookings`, `subscriptions`, `transactions`, `bom`, `date_dim`, `salary_history`, etc.). Those are introduced inline with the query and follow the obvious column shapes named in the problem text.

**How to use this to learn a query you can't write:** (1) read the prompt, (2) find the rows above it touches, (3) hand-compute the answer on paper, (4) *then* write SQL and check it matches your hand computation. Working backward from "I know the answer is these rows, what produces them?" is how you build query intuition. The appendix at the bottom does this for all 12 boss-tier problems.

---

# TIER 1 — Basics: SELECT / WHERE / ORDER / LIMIT / DISTINCT

### 1. Select all rows and columns
```sql
SELECT * FROM employees;
```
**Twin:** Return every column from `customers`.
```sql
SELECT * FROM customers;
```

### 2. Select specific columns
```sql
SELECT emp_name, salary FROM employees;
```
**Twin:** Return `product_name` and `price` from `products`.
```sql
SELECT product_name, price FROM products;
```

### 3. Filter with WHERE (equality)
```sql
SELECT emp_name FROM employees WHERE dept_id = 3;
```
**Twin:** Customers from `'US'`.
```sql
SELECT customer_name FROM customers WHERE country = 'US';
```

### 4. Numeric comparison
```sql
SELECT emp_name, salary FROM employees WHERE salary > 100000;
```
**Twin:** Products priced under 20.
```sql
SELECT product_name FROM products WHERE price < 20;
```

### 5. Combine conditions (AND / OR)
```sql
SELECT emp_name FROM employees
WHERE dept_id = 3 AND salary > 90000;
```
**Twin:** Orders that are `'shipped'` OR `'delivered'`.
```sql
SELECT order_id FROM orders
WHERE status = 'shipped' OR status = 'delivered';
```

### 6. Order results
```sql
SELECT emp_name, salary FROM employees ORDER BY salary DESC;
```
**Twin:** Products by price ascending.
```sql
SELECT product_name, price FROM products ORDER BY price ASC;
```

### 7. Order by multiple keys
```sql
SELECT emp_name, dept_id, salary FROM employees
ORDER BY dept_id ASC, salary DESC;
```
**Twin:** Orders by customer, then most recent date first.
```sql
SELECT * FROM orders ORDER BY customer_id ASC, order_date DESC;
```

### 8. Limit rows (top-N by sort)
```sql
SELECT emp_name, salary FROM employees ORDER BY salary DESC LIMIT 5;
```
**Twin:** 10 most recent orders.
```sql
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;
```

### 9. Distinct values
```sql
SELECT DISTINCT country FROM customers;
```
**Twin:** Distinct product categories.
```sql
SELECT DISTINCT category FROM products;
```

### 10. Distinct combinations
```sql
SELECT DISTINCT dept_id, country FROM employees;
```
**Twin:** Distinct (customer, status) pairs in orders.
```sql
SELECT DISTINCT customer_id, status FROM orders;
```

---

# TIER 2 — Filtering operators: IN / BETWEEN / LIKE / NULL

### 11. IN list
```sql
SELECT emp_name FROM employees WHERE dept_id IN (1, 2, 5);
```
**Twin:** Orders with status in a set.
```sql
SELECT order_id FROM orders WHERE status IN ('pending', 'cancelled');
```

### 12. NOT IN
```sql
SELECT emp_name FROM employees WHERE dept_id NOT IN (1, 2);
```
**Twin:** Products not in given categories.
```sql
SELECT product_name FROM products WHERE category NOT IN ('toys', 'books');
```

### 13. BETWEEN (inclusive range)
```sql
SELECT emp_name, salary FROM employees WHERE salary BETWEEN 50000 AND 80000;
```
**Twin:** Orders in a date range.
```sql
SELECT * FROM orders WHERE order_date BETWEEN '2026-01-01' AND '2026-03-31';
```

### 14. LIKE (prefix match)
```sql
SELECT emp_name FROM employees WHERE emp_name LIKE 'A%';
```
**Twin:** Products whose name ends in `'Pro'`.
```sql
SELECT product_name FROM products WHERE product_name LIKE '%Pro';
```

### 15. LIKE (contains / single-char)
```sql
SELECT customer_name FROM customers WHERE customer_name LIKE '%son%';
```
**Twin:** Names with 'a' as the second letter.
```sql
SELECT emp_name FROM employees WHERE emp_name LIKE '_a%';
```

### 16. IS NULL
```sql
SELECT emp_name FROM employees WHERE manager_id IS NULL;
```
**Twin:** Customers with no recorded country.
```sql
SELECT customer_name FROM customers WHERE country IS NULL;
```

### 17. IS NOT NULL
```sql
SELECT emp_name FROM employees WHERE manager_id IS NOT NULL;
```
**Twin:** Orders with a non-null amount.
```sql
SELECT order_id FROM orders WHERE amount IS NOT NULL;
```

### 18. Operator precedence (parenthesize OR with AND)
```sql
SELECT emp_name FROM employees
WHERE dept_id = 3 AND (salary > 90000 OR country = 'US');
```
**Twin:** Shipped orders that are either large or recent.
```sql
SELECT order_id FROM orders
WHERE status = 'shipped' AND (amount > 500 OR order_date >= '2026-06-01');
```

---

# TIER 3 — Aggregation: COUNT / SUM / AVG, GROUP BY, HAVING

### 19. Count rows
```sql
SELECT COUNT(*) FROM employees;
```
**Twin:** Number of orders.
```sql
SELECT COUNT(*) FROM orders;
```

### 20. COUNT(col) vs COUNT(*) (NULL-aware)
```sql
SELECT COUNT(*) AS all_rows, COUNT(manager_id) AS with_manager
FROM employees;
```
**Twin:** Total customers vs. customers with a known country.
```sql
SELECT COUNT(*) AS total, COUNT(country) AS known_country FROM customers;
```

### 21. COUNT DISTINCT
```sql
SELECT COUNT(DISTINCT dept_id) FROM employees;
```
**Twin:** Distinct customers who ordered.
```sql
SELECT COUNT(DISTINCT customer_id) FROM orders;
```

### 22. SUM / AVG / MIN / MAX
```sql
SELECT SUM(amount) AS total, AVG(amount) AS mean,
       MIN(amount) AS lo, MAX(amount) AS hi
FROM orders;
```
**Twin:** Salary stats over employees.
```sql
SELECT SUM(salary), AVG(salary), MIN(salary), MAX(salary) FROM employees;
```

### 23. GROUP BY one column
```sql
SELECT dept_id, COUNT(*) AS headcount
FROM employees GROUP BY dept_id;
```
**Twin:** Order count per status.
```sql
SELECT status, COUNT(*) FROM orders GROUP BY status;
```

### 24. GROUP BY with aggregate
```sql
SELECT dept_id, AVG(salary) AS avg_salary
FROM employees GROUP BY dept_id;
```
**Twin:** Total revenue per product.
```sql
SELECT product_id, SUM(amount) AS revenue FROM orders GROUP BY product_id;
```

### 25. GROUP BY multiple columns
```sql
SELECT dept_id, country, COUNT(*) AS cnt
FROM employees GROUP BY dept_id, country;
```
**Twin:** Revenue per (customer, status).
```sql
SELECT customer_id, status, SUM(amount) FROM orders GROUP BY customer_id, status;
```

### 26. HAVING (filter groups)
```sql
SELECT dept_id, COUNT(*) AS headcount
FROM employees GROUP BY dept_id HAVING COUNT(*) > 10;
```
**Twin:** Customers with more than 5 orders.
```sql
SELECT customer_id, COUNT(*) AS orders
FROM orders GROUP BY customer_id HAVING COUNT(*) > 5;
```

### 27. WHERE + GROUP BY + HAVING together
```sql
SELECT dept_id, AVG(salary) AS avg_salary
FROM employees
WHERE country = 'US'
GROUP BY dept_id
HAVING AVG(salary) > 80000;
```
**Twin:** Among shipped orders, products with revenue over 10k.
```sql
SELECT product_id, SUM(amount) AS rev
FROM orders WHERE status = 'shipped'
GROUP BY product_id HAVING SUM(amount) > 10000;
```

### 28. Conditional aggregation (SUM of a CASE)
```sql
SELECT dept_id,
       SUM(CASE WHEN salary > 100000 THEN 1 ELSE 0 END) AS high_earners,
       COUNT(*) AS total
FROM employees GROUP BY dept_id;
```
**Twin:** Per customer, count shipped vs. cancelled orders.
```sql
SELECT customer_id,
       SUM(CASE WHEN status = 'shipped'   THEN 1 ELSE 0 END) AS shipped,
       SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders GROUP BY customer_id;
```

### 29. Ratio / percentage via conditional aggregation
```sql
SELECT dept_id,
       AVG(CASE WHEN salary > 100000 THEN 1.0 ELSE 0.0 END) AS pct_high
FROM employees GROUP BY dept_id;
```
**Twin:** Cancellation rate per product.
```sql
SELECT product_id,
       AVG(CASE WHEN status = 'cancelled' THEN 1.0 ELSE 0.0 END) AS cancel_rate
FROM orders GROUP BY product_id;
```

### 30. GROUP BY ordered by aggregate
```sql
SELECT product_id, SUM(amount) AS rev
FROM orders GROUP BY product_id ORDER BY rev DESC LIMIT 10;
```
**Twin:** Top 5 departments by headcount.
```sql
SELECT dept_id, COUNT(*) AS c FROM employees
GROUP BY dept_id ORDER BY c DESC LIMIT 5;
```

### 31. Aggregate over expression
```sql
SELECT SUM(quantity * 1.0) AS units, SUM(amount) / SUM(quantity) AS avg_unit_price
FROM orders;
```
**Twin:** Average order value (amount per order) per customer.
```sql
SELECT customer_id, SUM(amount) / COUNT(*) AS aov FROM orders GROUP BY customer_id;
```

### 32. GROUP BY with date bucket
```sql
SELECT DATE_TRUNC('month', order_date) AS mth, SUM(amount) AS rev
FROM orders GROUP BY DATE_TRUNC('month', order_date) ORDER BY mth;
-- Hive: GROUP BY TRUNC(order_date,'MM')  or  SUBSTR(CAST(order_date AS STRING),1,7)
```
**Twin:** Daily signup counts.
```sql
SELECT signup_date, COUNT(*) FROM customers GROUP BY signup_date ORDER BY signup_date;
```

---

# TIER 4 — Joins

### 33. INNER JOIN
```sql
SELECT e.emp_name, d.dept_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id;
```
**Twin:** Orders with their customer names.
```sql
SELECT o.order_id, c.customer_name
FROM orders o JOIN customers c ON o.customer_id = c.customer_id;
```

### 34. LEFT JOIN (keep all left rows)
```sql
SELECT e.emp_name, d.dept_name
FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id;
```
**Twin:** All customers and their orders (customers with none still appear).
```sql
SELECT c.customer_name, o.order_id
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

### 35. Find non-matches (anti-join via LEFT JOIN + NULL)
```sql
SELECT c.customer_name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```
**Twin:** Departments with no employees.
```sql
SELECT d.dept_name
FROM departments d
LEFT JOIN employees e ON d.dept_id = e.dept_id
WHERE e.emp_id IS NULL;
```

### 36. RIGHT JOIN
```sql
SELECT e.emp_name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept_id = d.dept_id;
```
**Twin:** Right join orders to products to keep all products.
```sql
SELECT o.order_id, p.product_name
FROM orders o RIGHT JOIN products p ON o.product_id = p.product_id;
```

### 37. FULL OUTER JOIN
```sql
SELECT e.emp_name, d.dept_name
FROM employees e
FULL OUTER JOIN departments d ON e.dept_id = d.dept_id;
```
**Twin:** Full outer join customers and orders.
```sql
SELECT c.customer_name, o.order_id
FROM customers c FULL OUTER JOIN orders o ON c.customer_id = o.customer_id;
```

### 38. Self-join (employees to their manager)
```sql
SELECT e.emp_name AS employee, m.emp_name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.emp_id;
```
**Twin:** Pairs of customers from the same country (no self-pairs, no dupes).
```sql
SELECT a.customer_name, b.customer_name
FROM customers a
JOIN customers b ON a.country = b.country AND a.customer_id < b.customer_id;
```

### 39. Multi-table join
```sql
SELECT o.order_id, c.customer_name, p.product_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id;
```
**Twin:** Employee name, department name, and location.
```sql
SELECT e.emp_name, d.dept_name, d.location
FROM employees e JOIN departments d ON e.dept_id = d.dept_id;
```

### 40. Aggregate after join
```sql
SELECT d.dept_name, COUNT(*) AS headcount, AVG(e.salary) AS avg_salary
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name;
```
**Twin:** Revenue per product category.
```sql
SELECT p.category, SUM(o.amount) AS rev
FROM orders o JOIN products p ON o.product_id = p.product_id
GROUP BY p.category;
```

### 41. Join + filter on the joined table
```sql
SELECT e.emp_name
FROM employees e
JOIN departments d ON e.dept_id = d.dept_id
WHERE d.location = 'Seattle';
```
**Twin:** Orders for products in the `'electronics'` category.
```sql
SELECT o.order_id
FROM orders o JOIN products p ON o.product_id = p.product_id
WHERE p.category = 'electronics';
```

### 42. LEFT JOIN with aggregate + COALESCE (count incl. zeros)
```sql
SELECT c.customer_name, COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_name;
```
**Twin:** Each department's revenue (0 if none), via left join.
```sql
SELECT d.dept_name, COALESCE(SUM(e.salary), 0) AS total_pay
FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_name;
```

### 43. Join on inequality / range
```sql
SELECT o.order_id, p.product_name
FROM orders o
JOIN products p ON o.product_id = p.product_id AND o.amount >= p.price;
```
**Twin:** Employees earning above their department's location-band floor (assume `departments.min_salary`).
```sql
SELECT e.emp_name
FROM employees e JOIN departments d
  ON e.dept_id = d.dept_id AND e.salary > d.min_salary;
```

### 44. CROSS JOIN (cartesian — use deliberately)
```sql
SELECT d.dept_name, cal.mth
FROM departments d
CROSS JOIN (SELECT DISTINCT DATE_TRUNC('month', order_date) AS mth FROM orders) cal;
```
**Twin:** All (category, country) combinations.
```sql
SELECT p.category, c.country
FROM (SELECT DISTINCT category FROM products) p
CROSS JOIN (SELECT DISTINCT country FROM customers) c;
```

### 45. Anti-join with NOT EXISTS
```sql
SELECT c.customer_name
FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```
**Twin:** Products never ordered, via NOT EXISTS.
```sql
SELECT p.product_name
FROM products p
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.product_id = p.product_id);
```

### 46. Semi-join with EXISTS (existence, not duplication)
```sql
SELECT c.customer_name
FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o
              WHERE o.customer_id = c.customer_id AND o.amount > 1000);
```
**Twin:** Departments that have at least one US employee.
```sql
SELECT d.dept_name FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e
              WHERE e.dept_id = d.dept_id AND e.country = 'US');
```

### 47. Join to a pre-aggregated subquery (avoid double counting)
```sql
SELECT c.customer_name, t.total
FROM customers c
JOIN (SELECT customer_id, SUM(amount) AS total
      FROM orders GROUP BY customer_id) t
  ON c.customer_id = t.customer_id;
```
**Twin:** Each department joined to its avg salary computed separately.
```sql
SELECT d.dept_name, a.avg_sal
FROM departments d
JOIN (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) a
  ON d.dept_id = a.dept_id;
```

### 48. USING and natural-key joins
```sql
SELECT emp_name, dept_name
FROM employees JOIN departments USING (dept_id);
```
**Twin:** Join orders and customers with USING.
```sql
SELECT order_id, customer_name FROM orders JOIN customers USING (customer_id);
```

---

# TIER 5 — Subqueries

### 49. Scalar subquery in WHERE (compare to global aggregate)
```sql
SELECT emp_name, salary FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```
**Twin:** Orders larger than the average order amount.
```sql
SELECT order_id FROM orders WHERE amount > (SELECT AVG(amount) FROM orders);
```

### 50. Subquery with IN
```sql
SELECT emp_name FROM employees
WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = 'Seattle');
```
**Twin:** Orders from customers in the US.
```sql
SELECT order_id FROM orders
WHERE customer_id IN (SELECT customer_id FROM customers WHERE country = 'US');
```

### 51. Correlated subquery (per-row)
```sql
SELECT e.emp_name, e.salary
FROM employees e
WHERE e.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e.dept_id);
```
**Twin:** Orders above the average amount for that same customer.
```sql
SELECT o.order_id FROM orders o
WHERE o.amount > (SELECT AVG(o2.amount) FROM orders o2 WHERE o2.customer_id = o.customer_id);
```

### 52. Scalar subquery in SELECT
```sql
SELECT emp_name, salary,
       salary - (SELECT AVG(salary) FROM employees) AS diff_from_mean
FROM employees;
```
**Twin:** Each order's amount minus the overall average.
```sql
SELECT order_id, amount - (SELECT AVG(amount) FROM orders) AS delta FROM orders;
```

### 53. Subquery in FROM (derived table)
```sql
SELECT dept_id, max_sal
FROM (SELECT dept_id, MAX(salary) AS max_sal FROM employees GROUP BY dept_id) t
WHERE max_sal > 120000;
```
**Twin:** Customers whose total spend exceeds 5000 (derived table).
```sql
SELECT customer_id, total FROM
  (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) s
WHERE total > 5000;
```

### 54. EXISTS vs IN with NULLs (prefer EXISTS / NOT EXISTS)
```sql
-- Safe even if subquery returns NULLs:
SELECT p.product_name FROM products p
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.product_id = p.product_id);
```
**Twin:** Customers who have never ordered (NOT EXISTS).
```sql
SELECT c.customer_name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```

### 55. ALL / ANY comparison
```sql
SELECT emp_name FROM employees
WHERE salary > ALL (SELECT salary FROM employees WHERE dept_id = 4);
```
**Twin:** Products priced above ANY product in `'books'`.
```sql
SELECT product_name FROM products
WHERE price > ANY (SELECT price FROM products WHERE category = 'books');
```

### 56. Subquery returning a max-date row (pre-window approach)
```sql
SELECT * FROM orders o
WHERE o.order_date = (SELECT MAX(order_date) FROM orders o2
                      WHERE o2.customer_id = o.customer_id);
```
**Twin:** Each employee's row at the company's max hire_date per dept.
```sql
SELECT * FROM employees e
WHERE e.hire_date = (SELECT MAX(hire_date) FROM employees e2 WHERE e2.dept_id = e.dept_id);
```

### 57. Nested subqueries (two levels)
```sql
SELECT emp_name FROM employees
WHERE dept_id IN (
  SELECT dept_id FROM departments
  WHERE location IN (SELECT location FROM departments GROUP BY location HAVING COUNT(*) > 1)
);
```
**Twin:** Orders for products in the most expensive category (by max price).
```sql
SELECT order_id FROM orders WHERE product_id IN (
  SELECT product_id FROM products WHERE category = (
    SELECT category FROM products GROUP BY category ORDER BY MAX(price) DESC LIMIT 1));
```

---

# TIER 6 — CTEs (Common Table Expressions)

### 58. Single CTE
```sql
WITH dept_avg AS (
  SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id
)
SELECT e.emp_name, e.salary, d.avg_sal
FROM employees e JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```
**Twin:** Customers spending above their country's average.
```sql
WITH country_avg AS (
  SELECT c.country, AVG(o.amount) AS avg_amt
  FROM customers c JOIN orders o ON c.customer_id = o.customer_id
  GROUP BY c.country)
SELECT o.order_id FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN country_avg ca ON c.country = ca.country
WHERE o.amount > ca.avg_amt;
```

### 59. Multiple CTEs
```sql
WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000),
seattle_depts AS (
  SELECT dept_id FROM departments WHERE location = 'Seattle')
SELECT h.emp_name FROM high_earners h
JOIN seattle_depts s ON h.dept_id = s.dept_id;
```
**Twin:** Big spenders who are also US customers.
```sql
WITH big AS (SELECT customer_id FROM orders GROUP BY customer_id HAVING SUM(amount) > 5000),
us AS (SELECT customer_id FROM customers WHERE country = 'US')
SELECT b.customer_id FROM big b JOIN us u ON b.customer_id = u.customer_id;
```

### 60. Chained CTEs (one feeds the next)
```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS mth, SUM(amount) AS rev
  FROM orders GROUP BY DATE_TRUNC('month', order_date)),
ranked AS (
  SELECT mth, rev, RANK() OVER (ORDER BY rev DESC) AS rnk FROM monthly)
SELECT * FROM ranked WHERE rnk <= 3;
```
**Twin:** Top 3 departments by total pay, chained.
```sql
WITH pay AS (SELECT dept_id, SUM(salary) AS total FROM employees GROUP BY dept_id),
r AS (SELECT dept_id, total, RANK() OVER (ORDER BY total DESC) rk FROM pay)
SELECT * FROM r WHERE rk <= 3;
```

### 61. Recursive CTE (org hierarchy)
```sql
WITH RECURSIVE chain AS (
  SELECT emp_id, emp_name, manager_id, 1 AS lvl
  FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.emp_name, e.manager_id, c.lvl + 1
  FROM employees e JOIN chain c ON e.manager_id = c.emp_id)
SELECT * FROM chain ORDER BY lvl;
```
**Twin:** Number series 1..10 via recursion.
```sql
WITH RECURSIVE nums AS (
  SELECT 1 AS n UNION ALL SELECT n + 1 FROM nums WHERE n < 10)
SELECT n FROM nums;
```
`-- Hive: no recursive CTE; use a numbers table or iterative jobs instead.`

### 62. Recursive: all reports under a manager
```sql
WITH RECURSIVE subtree AS (
  SELECT emp_id, emp_name, manager_id FROM employees WHERE emp_id = 42
  UNION ALL
  SELECT e.emp_id, e.emp_name, e.manager_id
  FROM employees e JOIN subtree s ON e.manager_id = s.emp_id)
SELECT * FROM subtree WHERE emp_id <> 42;
```
**Twin:** Full management chain *above* a given employee (walk up).
```sql
WITH RECURSIVE up AS (
  SELECT emp_id, emp_name, manager_id FROM employees WHERE emp_id = 99
  UNION ALL
  SELECT e.emp_id, e.emp_name, e.manager_id
  FROM employees e JOIN up u ON e.emp_id = u.manager_id)
SELECT * FROM up;
```

---

# TIER 7 — CASE, conditional logic, NULL handling

### 63. CASE in SELECT (bucketing)
```sql
SELECT emp_name,
  CASE WHEN salary >= 120000 THEN 'high'
       WHEN salary >= 70000  THEN 'mid'
       ELSE 'low' END AS band
FROM employees;
```
**Twin:** Label orders by size.
```sql
SELECT order_id,
  CASE WHEN amount > 1000 THEN 'large'
       WHEN amount > 100  THEN 'medium' ELSE 'small' END AS size_band
FROM orders;
```

### 64. Pivot with conditional aggregation
```sql
SELECT dept_id,
  SUM(CASE WHEN country='US' THEN 1 ELSE 0 END) AS us_cnt,
  SUM(CASE WHEN country='IN' THEN 1 ELSE 0 END) AS in_cnt
FROM employees GROUP BY dept_id;
```
**Twin:** Per product, count orders by status as columns.
```sql
SELECT product_id,
  SUM(CASE WHEN status='shipped'   THEN 1 ELSE 0 END) AS shipped,
  SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders GROUP BY product_id;
```

### 65. COALESCE (first non-null)
```sql
SELECT emp_name, COALESCE(manager_id, 0) AS mgr FROM employees;
```
**Twin:** Default missing country to `'unknown'`.
```sql
SELECT customer_name, COALESCE(country, 'unknown') AS country FROM customers;
```

### 66. NULLIF (avoid divide-by-zero)
```sql
SELECT product_id, SUM(amount) / NULLIF(SUM(quantity), 0) AS unit_price
FROM orders GROUP BY product_id;
```
**Twin:** Conversion ratio guarding a zero denominator.
```sql
SELECT creator_id,
  COUNT(CASE WHEN event_type='purchase' THEN 1 END) * 1.0
  / NULLIF(COUNT(CASE WHEN event_type='view' THEN 1 END), 0) AS conv
FROM video_events GROUP BY creator_id;
```

### 67. Nested CASE / ordered evaluation
```sql
SELECT order_id,
  CASE WHEN status='cancelled' THEN 'lost'
       WHEN amount > 1000 AND status='shipped' THEN 'vip_won'
       ELSE 'normal' END AS tag
FROM orders;
```
**Twin:** Tenure tiers from hire_date.
```sql
SELECT emp_name,
  CASE WHEN hire_date <= '2020-01-01' THEN 'veteran'
       WHEN hire_date <= '2024-01-01' THEN 'established'
       ELSE 'new' END AS tenure
FROM employees;
```

### 68. CASE inside ORDER BY (custom sort)
```sql
SELECT order_id, status FROM orders
ORDER BY CASE status WHEN 'pending' THEN 1 WHEN 'shipped' THEN 2 ELSE 3 END;
```
**Twin:** Sort employees US-first, then by name.
```sql
SELECT emp_name, country FROM employees
ORDER BY CASE WHEN country='US' THEN 0 ELSE 1 END, emp_name;
```

### 69. Boolean flags via CASE for QC
```sql
SELECT order_id,
  CASE WHEN amount IS NULL OR amount < 0 THEN 1 ELSE 0 END AS bad_amount
FROM orders;
```
**Twin:** Flag employees missing a manager or department.
```sql
SELECT emp_id,
  CASE WHEN manager_id IS NULL OR dept_id IS NULL THEN 1 ELSE 0 END AS incomplete
FROM employees;
```

### 70. Two-sided NULL-safe comparison
```sql
-- rows where country differs, treating NULL as a value
SELECT * FROM customers c1 JOIN customers c2 ON c1.customer_id < c2.customer_id
WHERE c1.country IS DISTINCT FROM c2.country;
-- Hive/Spark: use  NOT (a <=> b)
```
**Twin:** Find employees whose country differs from their manager's, NULL-safe.
```sql
SELECT e.emp_name FROM employees e JOIN employees m ON e.manager_id = m.emp_id
WHERE e.country IS DISTINCT FROM m.country;
```

---

# TIER 8 — String, Date/Time, and Regex

### 71. Concatenate
```sql
SELECT emp_name || ' (' || country || ')' AS label FROM employees;
-- Hive/Spark: CONCAT(emp_name, ' (', country, ')')
```
**Twin:** Build a "name - category" label for products.
```sql
SELECT CONCAT(product_name, ' - ', category) AS label FROM products;
```

### 72. UPPER / LOWER / TRIM
```sql
SELECT UPPER(TRIM(customer_name)) AS norm_name FROM customers;
```
**Twin:** Lowercase, trimmed department names.
```sql
SELECT LOWER(TRIM(dept_name)) FROM departments;
```

### 73. SUBSTRING / LENGTH
```sql
SELECT emp_name, SUBSTRING(emp_name, 1, 3) AS first3, LENGTH(emp_name) AS len
FROM employees;
```
**Twin:** First 7 chars of product name and its length.
```sql
SELECT SUBSTRING(product_name, 1, 7), LENGTH(product_name) FROM products;
```

### 74. Split / element extraction
```sql
-- Spark/Hive: SPLIT then index
SELECT SPLIT(emp_name, ' ')[0] AS first_name FROM employees;
-- Postgres: SPLIT_PART(emp_name, ' ', 1)
```
**Twin:** Domain from an email column (assume `customers.email`).
```sql
SELECT SPLIT(email, '@')[1] AS domain FROM customers;  -- Spark
-- Postgres: SPLIT_PART(email,'@',2)
```

### 75. REPLACE
```sql
SELECT REPLACE(customer_name, '  ', ' ') AS cleaned FROM customers;
```
**Twin:** Strip dashes from a phone column (`customers.phone`).
```sql
SELECT REPLACE(phone, '-', '') FROM customers;
```

### 76. Regex match (filter)
```sql
-- Spark/Hive
SELECT customer_name FROM customers WHERE customer_name RLIKE '^[A-C]';
-- Postgres: WHERE customer_name ~ '^[A-C]'
```
**Twin:** Orders whose status matches a regex of allowed words.
```sql
SELECT order_id FROM orders WHERE status RLIKE '^(shipped|delivered)$';
```

### 77. Regex extract
```sql
SELECT REGEXP_EXTRACT(customer_response, '[0-9]{3}-[0-9]{3}-[0-9]{4}', 0) AS phone
FROM calls;  -- Spark/Hive
```
**Twin:** Extract a 5-digit zip from an address column.
```sql
SELECT REGEXP_EXTRACT(address, '[0-9]{5}', 0) AS zip FROM customers;
```

### 78. Current date / date arithmetic
```sql
SELECT order_id, DATEDIFF(CURRENT_DATE, order_date) AS days_ago FROM orders;
-- Postgres: CURRENT_DATE - order_date
```
**Twin:** Tenure in days for each employee.
```sql
SELECT emp_name, DATEDIFF(CURRENT_DATE, hire_date) AS tenure_days FROM employees;
```

### 79. Extract parts (year/month/dow)
```sql
SELECT YEAR(order_date) AS yr, MONTH(order_date) AS mo, SUM(amount) AS rev
FROM orders GROUP BY YEAR(order_date), MONTH(order_date);
```
**Twin:** Signups by year.
```sql
SELECT YEAR(signup_date) AS yr, COUNT(*) FROM customers GROUP BY YEAR(signup_date);
```

### 80. Date add / truncation to period
```sql
SELECT order_id, DATE_ADD(order_date, 30) AS due_date FROM orders;  -- Spark/Hive
-- Postgres: order_date + INTERVAL '30 days'
```
**Twin:** Add 7 days to each session's login as an expiry.
```sql
SELECT session_id, DATE_ADD(CAST(login_time AS DATE), 7) AS expires FROM sessions;
```

### 81. Filter "last N days"
```sql
SELECT * FROM orders WHERE order_date >= DATE_SUB(CURRENT_DATE, 30);
```
**Twin:** Events in the last 7 days.
```sql
SELECT * FROM video_events WHERE event_time >= DATE_SUB(CURRENT_DATE, 7);
```

### 82. Count phone numbers in free text (regex count)
```sql
SELECT employee,
       SUM(SIZE(REGEXP_EXTRACT_ALL(customer_response, '[0-9]{3}-[0-9]{3}-[0-9]{4}'))) AS cnt
FROM calls GROUP BY employee ORDER BY cnt DESC LIMIT 10;  -- Spark 3.1+
-- Redshift/Oracle: REGEXP_COUNT(customer_response, '...')
```
**Twin:** Count hashtags (`#word`) per caption (`video_events.caption`).
```sql
SELECT video_id, SIZE(REGEXP_EXTRACT_ALL(caption, '#[A-Za-z0-9_]+')) AS tags
FROM video_events;
```

---

# TIER 9 — Set operations, deduplication, data-quality checks

### 83. UNION (distinct) vs UNION ALL
```sql
SELECT customer_id FROM orders WHERE status='shipped'
UNION
SELECT customer_id FROM orders WHERE amount > 1000;
```
**Twin:** All countries appearing in either customers or employees (deduped).
```sql
SELECT country FROM customers UNION SELECT country FROM employees;
```

### 84. UNION ALL (keep duplicates, faster)
```sql
SELECT 'order' AS src, order_date AS d FROM orders
UNION ALL
SELECT 'signup', signup_date FROM customers;
```
**Twin:** Stack two event sources into one feed.
```sql
SELECT user_id, login_time AS ts FROM sessions
UNION ALL
SELECT user_id, event_time FROM video_events;
```

### 85. INTERSECT
```sql
SELECT customer_id FROM orders WHERE status='shipped'
INTERSECT
SELECT customer_id FROM orders WHERE status='cancelled';
-- Hive: emulate with INNER JOIN on DISTINCT sets
```
**Twin:** Countries present in BOTH employees and customers.
```sql
SELECT country FROM employees INTERSECT SELECT country FROM customers;
```

### 86. EXCEPT / MINUS
```sql
SELECT customer_id FROM customers
EXCEPT
SELECT customer_id FROM orders;
-- Hive: LEFT JOIN ... WHERE right IS NULL
```
**Twin:** Products that exist but were never ordered, via EXCEPT.
```sql
SELECT product_id FROM products EXCEPT SELECT product_id FROM orders;
```

### 87. Find duplicates
```sql
SELECT customer_id, product_id, order_date, COUNT(*) AS c
FROM orders GROUP BY customer_id, product_id, order_date HAVING COUNT(*) > 1;
```
**Twin:** Duplicate employee names.
```sql
SELECT emp_name, COUNT(*) FROM employees GROUP BY emp_name HAVING COUNT(*) > 1;
```

### 88. Deduplicate keeping one row (ROW_NUMBER)
```sql
WITH r AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_date DESC) AS rn
  FROM orders)
SELECT * FROM r WHERE rn = 1;
```
**Twin:** One row per (user_id, video_id), keeping the latest event.
```sql
WITH r AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY event_time DESC) rn
  FROM video_events)
SELECT * FROM r WHERE rn = 1;
```

### 89. Data-quality: orphan foreign keys
```sql
SELECT o.order_id FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL;
```
**Twin:** Employees pointing to a non-existent department.
```sql
SELECT e.emp_id FROM employees e
LEFT JOIN departments d ON e.dept_id = d.dept_id WHERE d.dept_id IS NULL;
```

### 90. Data-quality: null / range / freshness audit
```sql
SELECT
  COUNT(*) AS rows,
  SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amount,
  SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS negative_amount,
  MAX(order_date) AS latest_load
FROM orders;
```
**Twin:** Audit `video_events` for null users and stale data.
```sql
SELECT COUNT(*) AS rows,
  SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) AS null_user,
  MAX(event_time) AS latest
FROM video_events;
```

---

# TIER 10 — Window functions

### 91. ROW_NUMBER
```sql
SELECT emp_name, dept_id, salary,
       ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
FROM employees;
```
**Twin:** Number each customer's orders newest-first.
```sql
SELECT order_id, customer_id,
       ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn
FROM orders;
```

### 92. RANK vs DENSE_RANK (ties)
```sql
SELECT emp_name, salary,
       RANK()       OVER (ORDER BY salary DESC) AS rnk,
       DENSE_RANK() OVER (ORDER BY salary DESC) AS dense
FROM employees;
```
**Twin:** Rank products by price with both functions.
```sql
SELECT product_name, price,
       RANK() OVER (ORDER BY price DESC) AS r,
       DENSE_RANK() OVER (ORDER BY price DESC) AS dr
FROM products;
```

### 93. Top-N per group (the classic)
```sql
WITH r AS (
  SELECT emp_name, dept_id, salary,
         ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
  FROM employees)
SELECT * FROM r WHERE rn <= 3;
```
**Twin:** Top 2 products by revenue within each category.
```sql
WITH rev AS (
  SELECT p.category, p.product_id, SUM(o.amount) AS r
  FROM orders o JOIN products p ON o.product_id=p.product_id
  GROUP BY p.category, p.product_id),
ranked AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY r DESC) rn FROM rev)
SELECT * FROM ranked WHERE rn <= 2;
```

### 94. Partitioned aggregate (window, no collapse)
```sql
SELECT emp_name, dept_id, salary,
       AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
```
**Twin:** Each order's amount alongside its customer's total.
```sql
SELECT order_id, customer_id, amount,
       SUM(amount) OVER (PARTITION BY customer_id) AS customer_total
FROM orders;
```

### 95. Percent of total (window)
```sql
SELECT product_id, SUM(amount) AS rev,
       SUM(amount) * 100.0 / SUM(SUM(amount)) OVER () AS pct_of_total
FROM orders GROUP BY product_id;
```
**Twin:** Each department's share of total payroll.
```sql
SELECT dept_id, SUM(salary) AS pay,
       SUM(salary) * 100.0 / SUM(SUM(salary)) OVER () AS pct
FROM employees GROUP BY dept_id;
```

### 96. LAG (previous row)
```sql
WITH m AS (SELECT DATE_TRUNC('month',order_date) mth, SUM(amount) rev
           FROM orders GROUP BY DATE_TRUNC('month',order_date))
SELECT mth, rev, LAG(rev) OVER (ORDER BY mth) AS prev_rev FROM m;
```
**Twin:** Day-over-day session counts with previous day.
```sql
WITH d AS (SELECT CAST(login_time AS DATE) dt, COUNT(*) c FROM sessions
           GROUP BY CAST(login_time AS DATE))
SELECT dt, c, LAG(c) OVER (ORDER BY dt) AS prev FROM d;
```

### 97. LEAD (next row)
```sql
SELECT user_id, event_time,
       LEAD(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS next_event
FROM video_events;
```
**Twin:** Each order's next order date for the same customer.
```sql
SELECT order_id, customer_id, order_date,
       LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order
FROM orders;
```

### 98. Month-over-month growth %
```sql
WITH m AS (SELECT DATE_TRUNC('month',order_date) mth, SUM(amount) rev
           FROM orders GROUP BY DATE_TRUNC('month',order_date))
SELECT mth, rev,
  (rev - LAG(rev) OVER (ORDER BY mth)) * 100.0 / LAG(rev) OVER (ORDER BY mth) AS mom_pct
FROM m;
```
**Twin:** MoM growth in active users (distinct users per month).
```sql
WITH m AS (SELECT DATE_TRUNC('month',event_time) mth, COUNT(DISTINCT user_id) u
           FROM video_events GROUP BY DATE_TRUNC('month',event_time))
SELECT mth, u, (u - LAG(u) OVER (ORDER BY mth))*100.0/LAG(u) OVER (ORDER BY mth) AS pct
FROM m;
```

### 99. Running / cumulative total
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
       SUM(rev) OVER (ORDER BY order_date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM d;
```
**Twin:** Cumulative headcount by hire_date.
```sql
WITH h AS (SELECT hire_date, COUNT(*) c FROM employees GROUP BY hire_date)
SELECT hire_date, SUM(c) OVER (ORDER BY hire_date) AS cumulative FROM h;
```

### 100. Moving average (sliding frame)
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
       AVG(rev) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM d;
```
**Twin:** 3-day moving average of daily active users.
```sql
WITH d AS (SELECT CAST(event_time AS DATE) dt, COUNT(DISTINCT user_id) u
           FROM video_events GROUP BY CAST(event_time AS DATE))
SELECT dt, u, AVG(u) OVER (ORDER BY dt ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3
FROM d;
```

### 101. FIRST_VALUE / LAST_VALUE
```sql
SELECT emp_name, dept_id, salary,
  FIRST_VALUE(emp_name) OVER (PARTITION BY dept_id ORDER BY salary DESC) AS top_earner
FROM employees;
```
**Twin:** First product each customer ever ordered.
```sql
SELECT customer_id, order_date,
  FIRST_VALUE(product_id) OVER (PARTITION BY customer_id ORDER BY order_date) AS first_product
FROM orders;
```

### 102. NTILE (quartiles/deciles)
```sql
SELECT emp_name, salary, NTILE(4) OVER (ORDER BY salary) AS quartile
FROM employees;
```
**Twin:** Split customers into 10 deciles by total spend.
```sql
WITH s AS (SELECT customer_id, SUM(amount) t FROM orders GROUP BY customer_id)
SELECT customer_id, t, NTILE(10) OVER (ORDER BY t) AS decile FROM s;
```

### 103. PERCENT_RANK / CUME_DIST
```sql
SELECT emp_name, salary, PERCENT_RANK() OVER (ORDER BY salary) AS pr
FROM employees;
```
**Twin:** Percentile rank of each product by price.
```sql
SELECT product_name, price, PERCENT_RANK() OVER (ORDER BY price) AS pr FROM products;
```

### 104. Median via PERCENTILE_CONT
```sql
SELECT dept_id,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees GROUP BY dept_id;
-- Spark/Hive: PERCENTILE(salary, 0.5) or PERCENTILE_APPROX(salary, 0.5)
```
**Twin:** Median order amount per status.
```sql
SELECT status, PERCENTILE_APPROX(amount, 0.5) AS median_amt
FROM orders GROUP BY status;  -- Spark/Hive
```

### 105. Difference from running max (drawdown)
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
  MAX(rev) OVER (ORDER BY order_date ROWS UNBOUNDED PRECEDING) - rev AS below_peak
FROM d;
```
**Twin:** Each day's distance below the all-time-high DAU.
```sql
WITH d AS (SELECT CAST(event_time AS DATE) dt, COUNT(DISTINCT user_id) u
           FROM video_events GROUP BY CAST(event_time AS DATE))
SELECT dt, u, MAX(u) OVER (ORDER BY dt ROWS UNBOUNDED PRECEDING) - u AS below_peak FROM d;
```

### 106. Nth highest value (e.g., 3rd-highest salary)
```sql
WITH r AS (SELECT DISTINCT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr
           FROM employees)
SELECT salary FROM r WHERE dr = 3;
```
**Twin:** 2nd most expensive product per category.
```sql
WITH r AS (SELECT category, product_name, price,
             DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) dr FROM products)
SELECT * FROM r WHERE dr = 2;
```

### 107. Compare row to group min/max in window
```sql
SELECT emp_name, dept_id, salary,
  salary - MIN(salary) OVER (PARTITION BY dept_id) AS above_dept_min
FROM employees;
```
**Twin:** Order amount minus the customer's smallest order.
```sql
SELECT order_id, customer_id, amount,
  amount - MIN(amount) OVER (PARTITION BY customer_id) AS above_min
FROM orders;
```

### 108. Count within window (rolling distinct-ish / position)
```sql
SELECT user_id, event_time,
  COUNT(*) OVER (PARTITION BY user_id ORDER BY event_time
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS events_so_far
FROM video_events;
```
**Twin:** Cumulative number of orders per customer over time.
```sql
SELECT order_id, customer_id, order_date,
  COUNT(*) OVER (PARTITION BY customer_id ORDER BY order_date
                 ROWS UNBOUNDED PRECEDING) AS order_seq
FROM orders;
```

---

# TIER 11 — Advanced analytics patterns

### 109. Gaps & islands: consecutive-day streaks
```sql
WITH active AS (SELECT DISTINCT user_id, CAST(event_time AS DATE) AS d FROM video_events),
g AS (SELECT user_id, d,
        DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d)) AS grp
      FROM active)
SELECT user_id, MIN(d) AS streak_start, COUNT(*) AS streak_len
FROM g GROUP BY user_id, grp HAVING COUNT(*) >= 3;
```
**Twin:** Customers ordering 3+ days in a row.
```sql
WITH a AS (SELECT DISTINCT customer_id, order_date d FROM orders),
g AS (SELECT customer_id, d,
        DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY d)) grp FROM a)
SELECT customer_id, MIN(d), COUNT(*) FROM g GROUP BY customer_id, grp HAVING COUNT(*)>=3;
```

### 110. Sessionization (30-min inactivity gap → new session)
```sql
WITH e AS (
  SELECT user_id, event_time,
         LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_t
  FROM video_events),
flagged AS (
  SELECT user_id, event_time,
    CASE WHEN prev_t IS NULL
          OR (UNIX_TIMESTAMP(event_time)-UNIX_TIMESTAMP(prev_t)) > 1800
         THEN 1 ELSE 0 END AS new_session
  FROM e)
SELECT user_id, event_time,
  SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_time
                         ROWS UNBOUNDED PRECEDING) AS session_id
FROM flagged;
```
**Twin:** Sessionize orders where a >7-day gap starts a new "shopping spell".
```sql
WITH e AS (SELECT customer_id, order_date,
             LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) prev
           FROM orders),
f AS (SELECT customer_id, order_date,
        CASE WHEN prev IS NULL OR DATEDIFF(order_date, prev) > 7 THEN 1 ELSE 0 END ns
      FROM e)
SELECT customer_id, order_date,
  SUM(ns) OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS spell
FROM f;
```

### 111. Peak concurrency (sweep line)
```sql
WITH ev AS (
  SELECT login_time AS ts, 1 AS delta FROM sessions
  UNION ALL SELECT logout_time, -1 FROM sessions),
run AS (
  SELECT ts, SUM(delta) OVER (ORDER BY ts, delta DESC
                              ROWS UNBOUNDED PRECEDING) AS concurrent FROM ev)
SELECT ts, concurrent FROM run ORDER BY concurrent DESC LIMIT 1;
```
**Twin:** Peak simultaneous active video streams (assume `streams(start_ts, end_ts)`).
```sql
WITH ev AS (SELECT start_ts ts, 1 d FROM streams UNION ALL SELECT end_ts, -1 FROM streams),
run AS (SELECT ts, SUM(d) OVER (ORDER BY ts, d DESC ROWS UNBOUNDED PRECEDING) c FROM ev)
SELECT ts, c FROM run ORDER BY c DESC LIMIT 1;
```

### 112. Retention: day-1 retention rate
```sql
WITH first_day AS (
  SELECT user_id, MIN(CAST(event_time AS DATE)) AS d0 FROM video_events GROUP BY user_id),
returned AS (
  SELECT f.user_id FROM first_day f
  JOIN video_events v ON v.user_id=f.user_id
   AND CAST(v.event_time AS DATE) = DATE_ADD(f.d0, 1))
SELECT COUNT(DISTINCT r.user_id) * 1.0 / COUNT(DISTINCT f.user_id) AS d1_retention
FROM first_day f LEFT JOIN returned r ON f.user_id=r.user_id;
```
**Twin:** Day-1 retention for customers (first order date + 1).
```sql
WITH f AS (SELECT customer_id, MIN(order_date) d0 FROM orders GROUP BY customer_id),
ret AS (SELECT f.customer_id FROM f JOIN orders o ON o.customer_id=f.customer_id
        AND o.order_date = DATE_ADD(f.d0,1))
SELECT COUNT(DISTINCT ret.customer_id)*1.0/COUNT(DISTINCT f.customer_id)
FROM f LEFT JOIN ret ON f.customer_id=ret.customer_id;
```

### 113. Cohort table (signup month x activity month)
```sql
WITH cohort AS (
  SELECT customer_id, DATE_TRUNC('month', signup_date) AS cohort_m FROM customers),
act AS (
  SELECT customer_id, DATE_TRUNC('month', order_date) AS act_m FROM orders)
SELECT c.cohort_m, a.act_m, COUNT(DISTINCT a.customer_id) AS active
FROM cohort c JOIN act a ON c.customer_id=a.customer_id
GROUP BY c.cohort_m, a.act_m ORDER BY c.cohort_m, a.act_m;
```
**Twin:** Creator cohort by join month vs. month they posted (assume `video_events.creator_id`).
```sql
WITH ch AS (SELECT creator_id, DATE_TRUNC('month', join_date) cm FROM creators),
a AS (SELECT creator_id, DATE_TRUNC('month', event_time) am FROM video_events)
SELECT ch.cm, a.am, COUNT(DISTINCT a.creator_id)
FROM ch JOIN a ON ch.creator_id=a.creator_id GROUP BY ch.cm, a.am;
```

### 114. Funnel conversion (view → like → purchase)
```sql
SELECT
  COUNT(DISTINCT CASE WHEN event_type='view'     THEN user_id END) AS viewers,
  COUNT(DISTINCT CASE WHEN event_type='like'     THEN user_id END) AS likers,
  COUNT(DISTINCT CASE WHEN event_type='purchase' THEN user_id END) AS buyers
FROM video_events;
```
**Twin:** Order funnel: carted vs. shipped vs. delivered (assume status values).
```sql
SELECT
  COUNT(DISTINCT CASE WHEN status='carted'    THEN customer_id END) AS carted,
  COUNT(DISTINCT CASE WHEN status='shipped'   THEN customer_id END) AS shipped,
  COUNT(DISTINCT CASE WHEN status='delivered' THEN customer_id END) AS delivered
FROM orders;
```

### 115. Year-over-year comparison (self-join on year)
```sql
WITH y AS (SELECT YEAR(order_date) yr, SUM(amount) rev FROM orders GROUP BY YEAR(order_date))
SELECT t.yr, t.rev, p.rev AS prev_yr,
       (t.rev - p.rev)*100.0/p.rev AS yoy_pct
FROM y t LEFT JOIN y p ON t.yr = p.yr + 1;
```
**Twin:** YoY change in distinct active users.
```sql
WITH y AS (SELECT YEAR(event_time) yr, COUNT(DISTINCT user_id) u
           FROM video_events GROUP BY YEAR(event_time))
SELECT t.yr, t.u, (t.u - p.u)*100.0/p.u AS yoy
FROM y t LEFT JOIN y p ON t.yr = p.yr + 1;
```

### 116. Pivot rows to columns (manual)
```sql
SELECT customer_id,
  SUM(CASE WHEN YEAR(order_date)=2025 THEN amount ELSE 0 END) AS rev_2025,
  SUM(CASE WHEN YEAR(order_date)=2026 THEN amount ELSE 0 END) AS rev_2026
FROM orders GROUP BY customer_id;
```
**Twin:** Pivot event counts by type per creator.
```sql
SELECT creator_id,
  SUM(CASE WHEN event_type='view' THEN 1 ELSE 0 END) AS views,
  SUM(CASE WHEN event_type='like' THEN 1 ELSE 0 END) AS likes
FROM video_events GROUP BY creator_id;
```

### 117. Unpivot columns to rows
```sql
SELECT customer_id, 'q1' AS quarter, q1_rev AS rev FROM customer_quarter
UNION ALL SELECT customer_id, 'q2', q2_rev FROM customer_quarter;
-- Spark: STACK(2,'q1',q1_rev,'q2',q2_rev)
```
**Twin:** Unpivot a wide metrics table (`m(creator_id, views, likes)`).
```sql
SELECT creator_id, 'views' AS metric, views AS val FROM m
UNION ALL SELECT creator_id, 'likes', likes FROM m;
```

### 118. Median split / above-median flag
```sql
WITH med AS (SELECT PERCENTILE_APPROX(amount,0.5) AS m FROM orders)
SELECT o.order_id, CASE WHEN o.amount > med.m THEN 'above' ELSE 'below' END AS half
FROM orders o CROSS JOIN med;
```
**Twin:** Flag employees above the company median salary.
```sql
WITH med AS (SELECT PERCENTILE_APPROX(salary,0.5) m FROM employees)
SELECT e.emp_name, CASE WHEN e.salary > med.m THEN 'above' ELSE 'below' END
FROM employees e CROSS JOIN med;
```

### 119. First/last touch attribution
```sql
WITH ordered AS (
  SELECT user_id, event_type, event_time,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time)      AS first_rn,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time DESC) AS last_rn
  FROM video_events)
SELECT user_id,
  MAX(CASE WHEN first_rn=1 THEN event_type END) AS first_touch,
  MAX(CASE WHEN last_rn=1  THEN event_type END) AS last_touch
FROM ordered GROUP BY user_id;
```
**Twin:** First and last product per customer.
```sql
WITH o AS (SELECT customer_id, product_id, order_date,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) f,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) l FROM orders)
SELECT customer_id, MAX(CASE WHEN f=1 THEN product_id END) first_p,
       MAX(CASE WHEN l=1 THEN product_id END) last_p FROM o GROUP BY customer_id;
```

### 120. Time-to-event (gap between two milestones)
```sql
WITH s AS (
  SELECT user_id,
    MIN(CASE WHEN event_type='view'     THEN event_time END) AS first_view,
    MIN(CASE WHEN event_type='purchase' THEN event_time END) AS first_buy
  FROM video_events GROUP BY user_id)
SELECT user_id, DATEDIFF(first_buy, CAST(first_view AS DATE)) AS days_to_purchase
FROM s WHERE first_buy IS NOT NULL;
```
**Twin:** Days from signup to first order per customer.
```sql
WITH s AS (SELECT customer_id, signup_date,
             (SELECT MIN(order_date) FROM orders o WHERE o.customer_id=c.customer_id) fo
           FROM customers c)
SELECT customer_id, DATEDIFF(fo, signup_date) AS days_to_first_order FROM s WHERE fo IS NOT NULL;
```

---

# TIER 12 — Data-engineering / Hive-Spark specific

### 121. Incremental load (only new/changed rows)
```sql
INSERT INTO target_table
SELECT * FROM source_table s
WHERE s.updated_at > (SELECT COALESCE(MAX(updated_at), '1970-01-01') FROM target_table);
```
**Twin:** Load only orders newer than the warehouse high-water mark.
```sql
INSERT INTO dw_orders
SELECT * FROM orders WHERE order_date >
  (SELECT COALESCE(MAX(order_date),'1970-01-01') FROM dw_orders);
```

### 122. CDC dedup: latest version per key
```sql
WITH r AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY pk ORDER BY op_ts DESC) AS rn
  FROM cdc_stream)
SELECT * FROM r WHERE rn = 1 AND op_type <> 'DELETE';
```
**Twin:** Latest non-deleted creator profile from a change log.
```sql
WITH r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY creator_id ORDER BY change_ts DESC) rn
           FROM creator_changes)
SELECT * FROM r WHERE rn=1 AND op <> 'DELETE';
```

### 123. SCD Type 2: close old version, open new
```sql
-- mark current row as expired when a new version arrives
SELECT pk, attr, effective_from, effective_to, is_current
FROM dim_table
WHERE is_current = TRUE;
-- build new versions:
WITH changes AS (
  SELECT s.pk, s.attr, CURRENT_DATE AS effective_from
  FROM staging s JOIN dim_table d ON s.pk=d.pk AND d.is_current
  WHERE s.attr <> d.attr)
SELECT * FROM changes;  -- then INSERT new + UPDATE old.effective_to
```
**Twin:** Detect creators whose `region` changed vs. the current dim row.
```sql
SELECT s.creator_id, d.region AS old_region, s.region AS new_region
FROM staging_creators s JOIN dim_creator d
  ON s.creator_id=d.creator_id AND d.is_current
WHERE s.region <> d.region;
```

### 124. Partition pruning (filter the partition column)
```sql
SELECT * FROM events_partitioned
WHERE dt BETWEEN '2026-06-01' AND '2026-06-07';  -- dt is the partition key
```
**Twin:** Read only one day's partition of orders.
```sql
SELECT * FROM orders_partitioned WHERE dt = '2026-06-15';
```

### 125. Dynamic partition insert (Hive/Spark)
```sql
SET hive.exec.dynamic.partition.mode=nonstrict;
INSERT OVERWRITE TABLE orders_partitioned PARTITION (dt)
SELECT order_id, customer_id, amount, order_date AS dt FROM staging_orders;
```
**Twin:** Write events partitioned by event date.
```sql
INSERT OVERWRITE TABLE events_part PARTITION (dt)
SELECT event_id, user_id, event_type, CAST(event_time AS DATE) AS dt FROM staging_events;
```

### 126. explode an array column (Spark/Hive)
```sql
SELECT user_id, tag
FROM video_events
LATERAL VIEW EXPLODE(SPLIT(hashtags, ',')) t AS tag;
```
**Twin:** Explode a comma-separated `categories` column in products.
```sql
SELECT product_id, cat FROM products
LATERAL VIEW EXPLODE(SPLIT(categories, ',')) c AS cat;
```

### 127. collect_list / collect_set (aggregate into array)
```sql
SELECT customer_id, COLLECT_SET(product_id) AS products_bought
FROM orders GROUP BY customer_id;
```
**Twin:** All distinct event types per user as an array.
```sql
SELECT user_id, COLLECT_SET(event_type) FROM video_events GROUP BY user_id;
```

### 128. map / struct access
```sql
SELECT user_id, props['device'] AS device FROM events_with_map;
```
**Twin:** Access a nested struct field.
```sql
SELECT user_id, geo.country FROM events_with_struct;
```

### 129. Broadcast-join hint (small dim, large fact)
```sql
SELECT /*+ BROADCAST(d) */ o.order_id, d.dept_name
FROM big_orders o JOIN small_dim d ON o.dept_id = d.dept_id;  -- Spark hint
```
**Twin:** Broadcast a small creators dim against a huge events fact.
```sql
SELECT /*+ BROADCAST(c) */ v.event_id, c.creator_name
FROM video_events v JOIN creators c ON v.creator_id = c.creator_id;
```

### 130. QUALIFY (filter on window result, where supported)
```sql
SELECT emp_name, dept_id, salary
FROM employees
QUALIFY ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) <= 3;
-- Hive/Spark lacking QUALIFY: wrap in a subquery/CTE and filter on rn
```
**Twin:** Latest order per customer using QUALIFY.
```sql
SELECT * FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1;
```

### 131. Approx distinct count at scale
```sql
SELECT APPROX_COUNT_DISTINCT(user_id) AS approx_users FROM video_events;
-- Hive: COUNT(DISTINCT ...) or ndv()-style UDFs; Spark: approx_count_distinct
```
**Twin:** Approx distinct customers in orders.
```sql
SELECT APPROX_COUNT_DISTINCT(customer_id) FROM orders;
```

### 132. Bucketed sampling / TABLESAMPLE
```sql
SELECT * FROM video_events TABLESAMPLE (1 PERCENT);
```
**Twin:** Sample ~5% of orders for a quick profile.
```sql
SELECT * FROM orders TABLESAMPLE (5 PERCENT);
```

---

# Quick pattern cheat-sheet (memorize the trigger → tool mapping)
- "Top N per group" → `ROW_NUMBER()/RANK()` in a CTE, filter `rn <= N`
- "Latest/most-recent per key" → `ROW_NUMBER() ... ORDER BY ts DESC`, keep `rn=1`
- "Consecutive runs / streaks" → gaps & islands: `date - ROW_NUMBER()` as group key
- "Sessionize / split by inactivity gap" → `LAG` time diff → flag → running `SUM` of flags
- "Peak concurrent" → +1/-1 event union → running `SUM` ordered by time → `MAX`
- "Period-over-period growth" → `LAG`/`LEAD` or self-join on year/month
- "Running total / moving average" → window `SUM`/`AVG` with `ROWS BETWEEN ... `
- "Median/percentile" → `PERCENTILE_CONT` (ANSI) / `PERCENTILE_APPROX` (Spark/Hive)
- "Find non-matches" → `LEFT JOIN ... WHERE right IS NULL` or `NOT EXISTS`
- "Dedup" → `ROW_NUMBER()` per key, keep one; or `GROUP BY ... HAVING COUNT(*)>1` to detect
- "Filter on a window value" → `QUALIFY`, or wrap in CTE and filter `rn`
- "% of total" → `x / SUM(x) OVER ()`
- "Pivot" → `SUM(CASE WHEN cat=... THEN v END)`; "Unpivot" → `UNION ALL` / `STACK`
- Prefer `NOT EXISTS` over `NOT IN` when the subquery can return NULLs.
- Always confirm the **grain** before you write the query: one row per what?

**How to drill this doc:** work top-to-bottom one tier per sitting. Read the worked query, cover the Twin's solution, solve the Twin, then check. If both feel automatic, move on; if not, redo the worked one from scratch.

---

# TIER 13 — God tier (the ones that separate offers from rejections)
These reward a *mental pattern*, not syntax recall. For each, the comment names the trick. If you can derive the trick cold, you're past most DE bars.

### 133. Median without PERCENTILE (dual ROW_NUMBER trick)
```sql
-- Trick: a row is "middle" iff its rank from the top equals its rank from the bottom (±1).
WITH r AS (
  SELECT salary,
         ROW_NUMBER() OVER (ORDER BY salary)      AS asc_rn,
         COUNT(*)     OVER ()                      AS n
  FROM employees)
SELECT AVG(salary) AS median
FROM r
WHERE asc_rn IN ((n + 1) / 2, (n + 2) / 2);   -- integer division handles odd & even
```
**Twin:** Median order amount per status, no percentile function.
```sql
WITH r AS (
  SELECT status, amount,
         ROW_NUMBER() OVER (PARTITION BY status ORDER BY amount) rn,
         COUNT(*)     OVER (PARTITION BY status) n
  FROM orders)
SELECT status, AVG(amount) AS median FROM r
WHERE rn IN ((n+1)/2, (n+2)/2) GROUP BY status;
```

### 134. Longest consecutive-day streak per user (max island length)
```sql
-- Trick: gaps & islands → group consecutive dates by (date − row_number), then take the largest group.
WITH a AS (SELECT DISTINCT user_id, CAST(event_time AS DATE) d FROM video_events),
g AS (SELECT user_id, d,
        DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d)) AS grp FROM a),
islands AS (SELECT user_id, grp, COUNT(*) AS len FROM g GROUP BY user_id, grp)
SELECT user_id, MAX(len) AS longest_streak FROM islands GROUP BY user_id;
```
**Twin:** Longest run of consecutive ordering days per customer.
```sql
WITH a AS (SELECT DISTINCT customer_id, order_date d FROM orders),
g AS (SELECT customer_id, d, DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY d)) grp FROM a)
SELECT customer_id, MAX(c) FROM (SELECT customer_id, grp, COUNT(*) c FROM g GROUP BY customer_id, grp) t
GROUP BY customer_id;
```

### 135. Running total that RESETS on a flag
```sql
-- Trick: cumulative count of the reset flag defines the segment id; sum within segment.
WITH base AS (
  SELECT customer_id, order_date, amount, is_reset,
         SUM(is_reset) OVER (PARTITION BY customer_id ORDER BY order_date
                             ROWS UNBOUNDED PRECEDING) AS seg
  FROM orders)
SELECT customer_id, order_date, amount,
       SUM(amount) OVER (PARTITION BY customer_id, seg ORDER BY order_date
                         ROWS UNBOUNDED PRECEDING) AS running_since_reset
FROM base;   -- is_reset = 1 marks the first row of a new segment
```
**Twin:** Running event count per user that resets at the start of each calendar day.
```sql
WITH b AS (SELECT user_id, event_time, CAST(event_time AS DATE) dt FROM video_events)
SELECT user_id, event_time,
  COUNT(*) OVER (PARTITION BY user_id, dt ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS daily_seq
FROM b;
```

### 136. Pareto: smallest set of products making 80% of revenue
```sql
-- Trick: cumulative share sorted desc; keep rows until the running total first crosses 80%.
WITH rev AS (SELECT product_id, SUM(amount) r FROM orders GROUP BY product_id),
c AS (SELECT product_id, r,
        SUM(r) OVER (ORDER BY r DESC ROWS UNBOUNDED PRECEDING) AS running,
        SUM(r) OVER () AS total FROM rev)
SELECT product_id, r, running / total AS cum_share
FROM c
WHERE running - r < 0.8 * total;   -- includes the row that pushes past 80%
```
**Twin:** Top creators who together produce 90% of all views.
```sql
WITH v AS (SELECT creator_id, COUNT(*) c FROM video_events GROUP BY creator_id),
cc AS (SELECT creator_id, c, SUM(c) OVER (ORDER BY c DESC ROWS UNBOUNDED PRECEDING) run,
              SUM(c) OVER () tot FROM v)
SELECT creator_id FROM cc WHERE run - c < 0.9 * tot;
```

### 137. As-of join (point-in-time): latest price at-or-before each trade
```sql
-- Trick: join on quote_time <= trade_time, then ROW_NUMBER to keep the most recent.
WITH ranked AS (
  SELECT t.trade_id, t.symbol, t.trade_time, q.price, q.quote_time,
         ROW_NUMBER() OVER (PARTITION BY t.trade_id ORDER BY q.quote_time DESC) AS rn
  FROM trades t
  JOIN quotes q ON q.symbol = t.symbol AND q.quote_time <= t.trade_time)
SELECT trade_id, symbol, trade_time, price FROM ranked WHERE rn = 1;
```
**Twin:** For each purchase, the same user's most recent view before it (point-in-time feature).
```sql
WITH r AS (
  SELECT p.event_id, p.user_id, p.event_time buy_t, v.event_time view_t,
         ROW_NUMBER() OVER (PARTITION BY p.event_id ORDER BY v.event_time DESC) rn
  FROM video_events p
  JOIN video_events v ON v.user_id=p.user_id AND v.event_type='view' AND v.event_time <= p.event_time
  WHERE p.event_type='purchase')
SELECT event_id, user_id, buy_t, view_t FROM r WHERE rn=1;
```

### 138. Merge overlapping intervals (interval coalescing)
```sql
-- Trick: a new merged block starts when start_ts > running max(end_ts) of all prior rows.
WITH ordered AS (
  SELECT room_id, start_ts, end_ts,
    MAX(end_ts) OVER (PARTITION BY room_id ORDER BY start_ts
                      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
  FROM bookings),
grp AS (
  SELECT room_id, start_ts, end_ts,
    SUM(CASE WHEN prev_max_end IS NULL OR start_ts > prev_max_end THEN 1 ELSE 0 END)
      OVER (PARTITION BY room_id ORDER BY start_ts ROWS UNBOUNDED PRECEDING) AS g
  FROM ordered)
SELECT room_id, MIN(start_ts) AS merged_start, MAX(end_ts) AS merged_end
FROM grp GROUP BY room_id, g;
```
**Twin:** Merge contiguous/overlapping active subscription windows per user.
```sql
WITH o AS (SELECT user_id, start_date s, end_date e,
             MAX(end_date) OVER (PARTITION BY user_id ORDER BY start_date
                                 ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pe FROM subs),
g AS (SELECT user_id, s, e, SUM(CASE WHEN pe IS NULL OR s > pe THEN 1 ELSE 0 END)
        OVER (PARTITION BY user_id ORDER BY s ROWS UNBOUNDED PRECEDING) grp FROM o)
SELECT user_id, MIN(s), MAX(e) FROM g GROUP BY user_id, grp;
```

### 139. Detect double-booking (overlapping intervals)
```sql
-- Trick: two intervals overlap iff a.start < b.end AND b.start < a.end.
SELECT a.room_id, a.booking_id, b.booking_id
FROM bookings a JOIN bookings b
  ON a.room_id = b.room_id AND a.booking_id < b.booking_id
 AND a.start_ts < b.end_ts AND b.start_ts < a.end_ts;
```
**Twin:** Find employees with overlapping shifts.
```sql
SELECT a.emp_id, a.shift_id, b.shift_id FROM shifts a JOIN shifts b
  ON a.emp_id=b.emp_id AND a.shift_id<b.shift_id
 AND a.start_ts < b.end_ts AND b.start_ts < a.end_ts;
```

### 140. Recursive hierarchy path string ("CEO > VP > Mgr > you")
```sql
WITH RECURSIVE p AS (
  SELECT emp_id, emp_name, manager_id, CAST(emp_name AS VARCHAR(2000)) AS path, 1 AS depth
  FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.emp_name, e.manager_id, p.path || ' > ' || e.emp_name, p.depth + 1
  FROM employees e JOIN p ON e.manager_id = p.emp_id)
SELECT emp_id, depth, path FROM p;
-- Hive/Spark: no recursive CTE — iterate in Spark, or precompute paths in an upstream job.
```
**Twin:** Build a category breadcrumb path in a self-referencing categories table.
```sql
WITH RECURSIVE c AS (
  SELECT cat_id, name, parent_id, CAST(name AS VARCHAR(2000)) path FROM categories WHERE parent_id IS NULL
  UNION ALL
  SELECT k.cat_id, k.name, k.parent_id, c.path || ' / ' || k.name
  FROM categories k JOIN c ON k.parent_id = c.cat_id)
SELECT cat_id, path FROM c;
```

### 141. Recursive bill-of-materials quantity rollup
```sql
-- Trick: multiply quantities down the tree, then sum per leaf.
WITH RECURSIVE explode AS (
  SELECT parent_id, child_id, qty AS total_qty FROM bom WHERE parent_id = 'ASSEMBLY_1'
  UNION ALL
  SELECT b.parent_id, b.child_id, e.total_qty * b.qty
  FROM bom b JOIN explode e ON b.parent_id = e.child_id)
SELECT child_id, SUM(total_qty) AS qty_needed FROM explode GROUP BY child_id;
```
**Twin:** Total number of reports (all levels) under each top-level manager.
```sql
WITH RECURSIVE sub AS (
  SELECT emp_id AS root, emp_id FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT s.root, e.emp_id FROM employees e JOIN sub s ON e.manager_id = s.emp_id)
SELECT root, COUNT(*) - 1 AS total_reports FROM sub GROUP BY root;
```

### 142. Running DISTINCT count (windows can't COUNT DISTINCT)
```sql
-- Trick: count a value only on its FIRST occurrence, then take a running SUM of those flags.
WITH firsts AS (
  SELECT user_id, video_id, event_time,
         ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY event_time) AS occ
  FROM video_events)
SELECT user_id, event_time,
       SUM(CASE WHEN occ = 1 THEN 1 ELSE 0 END)
         OVER (PARTITION BY user_id ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS distinct_videos_so_far
FROM firsts;
```
**Twin:** Cumulative distinct products a customer has ever ordered, over time.
```sql
WITH f AS (SELECT customer_id, product_id, order_date,
             ROW_NUMBER() OVER (PARTITION BY customer_id, product_id ORDER BY order_date) occ FROM orders)
SELECT customer_id, order_date,
  SUM(CASE WHEN occ=1 THEN 1 ELSE 0 END)
    OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS distinct_products
FROM f;
```

### 143. Calendar-spine gap fill (zero-fill missing days)
```sql
-- Trick: drive off a complete date dimension with a LEFT JOIN; COALESCE the gaps.
WITH cal AS (SELECT d FROM date_dim WHERE d BETWEEN '2026-01-01' AND '2026-01-31'),
daily AS (SELECT order_date d, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT c.d, COALESCE(dy.rev, 0) AS rev
FROM cal c LEFT JOIN daily dy ON c.d = dy.d ORDER BY c.d;
```
**Twin:** Zero-filled daily active users, including days with zero activity.
```sql
WITH cal AS (SELECT d FROM date_dim WHERE d BETWEEN '2026-06-01' AND '2026-06-30'),
dau AS (SELECT CAST(event_time AS DATE) d, COUNT(DISTINCT user_id) u FROM video_events GROUP BY CAST(event_time AS DATE))
SELECT c.d, COALESCE(dau.u, 0) FROM cal c LEFT JOIN dau ON c.d = dau.d ORDER BY c.d;
```

### 144. Last-non-null carry-forward (forward fill)
```sql
-- Trick: cumulative COUNT of non-nulls forms a group id; the one non-null in each group is its value.
WITH s AS (
  SELECT user_id, event_time, status,
         COUNT(status) OVER (PARTITION BY user_id ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS grp
  FROM user_status)
SELECT user_id, event_time,
       MAX(status) OVER (PARTITION BY user_id, grp) AS status_filled
FROM s;
-- Where supported: LAST_VALUE(status) IGNORE NULLS OVER (... ) does this directly.
```
**Twin:** Forward-fill the last known price for each product across a daily timeline.
```sql
WITH s AS (SELECT product_id, dt, price,
             COUNT(price) OVER (PARTITION BY product_id ORDER BY dt ROWS UNBOUNDED PRECEDING) g FROM price_daily)
SELECT product_id, dt, MAX(price) OVER (PARTITION BY product_id, g) AS price_filled FROM s;
```

### 145. N-day retention curve (one query, all offsets)
```sql
WITH first_day AS (SELECT user_id, MIN(CAST(event_time AS DATE)) d0 FROM video_events GROUP BY user_id),
activity AS (SELECT DISTINCT user_id, CAST(event_time AS DATE) d FROM video_events)
SELECT DATEDIFF(a.d, f.d0) AS day_offset, COUNT(DISTINCT a.user_id) AS retained
FROM first_day f JOIN activity a ON a.user_id = f.user_id
WHERE DATEDIFF(a.d, f.d0) BETWEEN 0 AND 30
GROUP BY DATEDIFF(a.d, f.d0) ORDER BY day_offset;
```
**Twin:** Weekly order-retention curve (offset in weeks from first order).
```sql
WITH f AS (SELECT customer_id, MIN(order_date) d0 FROM orders GROUP BY customer_id),
a AS (SELECT DISTINCT customer_id, order_date d FROM orders)
SELECT FLOOR(DATEDIFF(a.d, f.d0)/7) AS week_off, COUNT(DISTINCT a.customer_id)
FROM f JOIN a ON a.customer_id=f.customer_id
WHERE DATEDIFF(a.d,f.d0) BETWEEN 0 AND 84 GROUP BY FLOOR(DATEDIFF(a.d,f.d0)/7);
```

### 146. Sequential funnel with enforced time ORDER (not just presence)
```sql
-- Trick: capture the MIN timestamp per step, then require step1_time < step2_time < step3_time.
WITH steps AS (
  SELECT user_id,
    MIN(CASE WHEN event_type='view'     THEN event_time END) AS t_view,
    MIN(CASE WHEN event_type='cart'     THEN event_time END) AS t_cart,
    MIN(CASE WHEN event_type='purchase' THEN event_time END) AS t_buy
  FROM video_events GROUP BY user_id)
SELECT
  SUM(CASE WHEN t_view IS NOT NULL THEN 1 ELSE 0 END)                                   AS viewed,
  SUM(CASE WHEN t_view < t_cart THEN 1 ELSE 0 END)                                      AS view_then_cart,
  SUM(CASE WHEN t_view < t_cart AND t_cart < t_buy THEN 1 ELSE 0 END)                   AS full_funnel
FROM steps;
```
**Twin:** Ordered funnel signup → first_order → repeat_order using timestamps.
```sql
WITH s AS (SELECT customer_id, signup_date td,
             (SELECT MIN(order_date) FROM orders o WHERE o.customer_id=c.customer_id) t1,
             (SELECT MIN(order_date) FROM orders o WHERE o.customer_id=c.customer_id
              AND order_date > (SELECT MIN(order_date) FROM orders o2 WHERE o2.customer_id=c.customer_id)) t2
           FROM customers c)
SELECT SUM(CASE WHEN t1 IS NOT NULL THEN 1 ELSE 0 END) ordered,
       SUM(CASE WHEN t2 IS NOT NULL THEN 1 ELSE 0 END) repeated FROM s;
```

### 147. Most frequent value per group with deterministic tie-break (mode)
```sql
WITH counts AS (SELECT dept_id, country, COUNT(*) c FROM employees GROUP BY dept_id, country),
ranked AS (SELECT dept_id, country, c,
             ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY c DESC, country ASC) rn FROM counts)
SELECT dept_id, country AS modal_country FROM ranked WHERE rn = 1;
```
**Twin:** Most-ordered product category per customer (break ties alphabetically).
```sql
WITH c AS (SELECT o.customer_id, p.category, COUNT(*) n FROM orders o JOIN products p ON o.product_id=p.product_id GROUP BY o.customer_id, p.category),
r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY n DESC, category) rn FROM c)
SELECT customer_id, category FROM r WHERE rn=1;
```

### 148. First negative running balance per account
```sql
WITH t AS (
  SELECT account_id, txn_time, amount,
         SUM(amount) OVER (PARTITION BY account_id ORDER BY txn_time ROWS UNBOUNDED PRECEDING) AS bal
  FROM transactions),
neg AS (SELECT account_id, txn_time, bal,
          ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY txn_time) rn
        FROM t WHERE bal < 0)
SELECT account_id, txn_time AS first_negative_at, bal FROM neg WHERE rn = 1;
```
**Twin:** First day cumulative inventory goes below zero (oversold) per SKU.
```sql
WITH t AS (SELECT sku, dt, delta,
             SUM(delta) OVER (PARTITION BY sku ORDER BY dt ROWS UNBOUNDED PRECEDING) onhand FROM inventory_moves),
n AS (SELECT sku, dt, onhand, ROW_NUMBER() OVER (PARTITION BY sku ORDER BY dt) rn FROM t WHERE onhand < 0)
SELECT sku, dt, onhand FROM n WHERE rn=1;
```

### 149. Longest monotonic-increasing run of a metric
```sql
-- Trick: flag where the increase breaks, cumulative-sum the breaks into run ids, take the longest run.
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date),
flagged AS (SELECT order_date, rev,
              CASE WHEN rev > LAG(rev) OVER (ORDER BY order_date) THEN 0 ELSE 1 END AS broke FROM d),
grp AS (SELECT order_date, rev,
          SUM(broke) OVER (ORDER BY order_date ROWS UNBOUNDED PRECEDING) g FROM flagged)
SELECT g, COUNT(*) run_len, MIN(order_date) start_d, MAX(order_date) end_d
FROM grp GROUP BY g ORDER BY run_len DESC LIMIT 1;
```
**Twin:** Longest increasing streak of daily active users.
```sql
WITH d AS (SELECT CAST(event_time AS DATE) dt, COUNT(DISTINCT user_id) u FROM video_events GROUP BY CAST(event_time AS DATE)),
f AS (SELECT dt, u, CASE WHEN u > LAG(u) OVER (ORDER BY dt) THEN 0 ELSE 1 END b FROM d),
g AS (SELECT dt, SUM(b) OVER (ORDER BY dt ROWS UNBOUNDED PRECEDING) grp FROM f)
SELECT grp, COUNT(*) len FROM g GROUP BY grp ORDER BY len DESC LIMIT 1;
```

### 150. Expand a date range into one row per day (recursive)
```sql
WITH RECURSIVE expanded AS (
  SELECT subscription_id, start_date AS d, end_date FROM subscriptions
  UNION ALL
  SELECT subscription_id, DATE_ADD(d, 1), end_date FROM expanded WHERE d < end_date)
SELECT subscription_id, d FROM expanded;
-- Hive/Spark: explode(sequence(start_date, end_date, interval 1 day)) instead.
```
**Twin:** Explode each employee's leave window into individual days.
```sql
WITH RECURSIVE x AS (
  SELECT emp_id, leave_start d, leave_end FROM leaves
  UNION ALL SELECT emp_id, DATE_ADD(d,1), leave_end FROM x WHERE d < leave_end)
SELECT emp_id, d FROM x;
```

### 151. Active EVERY day in a window (full coverage)
```sql
SELECT user_id
FROM video_events
WHERE event_time >= '2026-06-01' AND event_time < '2026-07-01'
GROUP BY user_id
HAVING COUNT(DISTINCT CAST(event_time AS DATE)) = 30;
```
**Twin:** Customers who ordered in every month of 2026 (all 12).
```sql
SELECT customer_id FROM orders WHERE YEAR(order_date)=2026
GROUP BY customer_id HAVING COUNT(DISTINCT MONTH(order_date)) = 12;
```

### 152. RANGE frame: true rolling 7 *days* (not 7 rows)
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
       SUM(rev) OVER (ORDER BY order_date
         RANGE BETWEEN INTERVAL '6' DAY PRECEDING AND CURRENT ROW) AS rolling_7d
FROM d;
-- Contrast: ROWS BETWEEN 6 PRECEDING counts 7 rows even if dates are missing → wrong window on sparse data.
```
**Twin:** Rolling 30-day revenue by calendar range.
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, SUM(rev) OVER (ORDER BY order_date
  RANGE BETWEEN INTERVAL '29' DAY PRECEDING AND CURRENT ROW) AS rolling_30d FROM d;
```

### 153. Concurrency histogram (active sessions per time bucket)
```sql
SELECT b.hr, COUNT(*) AS concurrent
FROM hour_dim b
JOIN sessions s ON s.login_time <= b.hr AND s.logout_time > b.hr
GROUP BY b.hr ORDER BY b.hr;
```
**Twin:** Open orders (placed but not shipped) per day from a status-history table.
```sql
SELECT c.d, COUNT(*) open_orders
FROM date_dim c JOIN orders o ON o.placed_date <= c.d AND (o.shipped_date IS NULL OR o.shipped_date > c.d)
GROUP BY c.d ORDER BY c.d;
```

### 154. Top-N per group INCLUDING all ties at the boundary
```sql
WITH r AS (SELECT dept_id, emp_name, salary,
             DENSE_RANK() OVER (PARTITION BY dept_id ORDER BY salary DESC) dr FROM employees)
SELECT * FROM r WHERE dr <= 3;   -- may return >3 rows per dept when salaries tie
```
**Twin:** All products tied within the two highest prices per category.
```sql
WITH r AS (SELECT category, product_name, price,
             DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) dr FROM products)
SELECT * FROM r WHERE dr <= 2;
```

### 155. Pivot + share + rank + cumulative share in one pass
```sql
WITH base AS (
  SELECT p.category, o.product_id, SUM(o.amount) rev
  FROM orders o JOIN products p ON o.product_id=p.product_id
  GROUP BY p.category, o.product_id)
SELECT category, product_id, rev,
  rev * 100.0 / SUM(rev) OVER (PARTITION BY category)                                  AS pct_of_category,
  RANK() OVER (PARTITION BY category ORDER BY rev DESC)                                AS rank_in_category,
  SUM(rev) OVER (PARTITION BY category ORDER BY rev DESC ROWS UNBOUNDED PRECEDING)
    * 100.0 / SUM(rev) OVER (PARTITION BY category)                                    AS cumulative_pct
FROM base;
```
**Twin:** Per-region creator views with share, rank, and cumulative share.
```sql
WITH b AS (SELECT c.region, v.creator_id, COUNT(*) views FROM video_events v JOIN creators c ON v.creator_id=c.creator_id GROUP BY c.region, v.creator_id)
SELECT region, creator_id, views,
  views*100.0/SUM(views) OVER (PARTITION BY region) pct,
  RANK() OVER (PARTITION BY region ORDER BY views DESC) rnk,
  SUM(views) OVER (PARTITION BY region ORDER BY views DESC ROWS UNBOUNDED PRECEDING)*100.0
    /SUM(views) OVER (PARTITION BY region) cum_pct
FROM b;
```

---

# TIER 14 — Query optimization & rewrite reasoning (senior-level "why", not just "what")
Interviewers escalate from "write it" to "now make it fast / why is it slow." Each item: the slow form, the fast rewrite, and the reason. Be able to *say the reason out loud*.

### 156. Correlated subquery → window function
```sql
-- SLOW: re-scans orders once per row (O(n^2) feel on big data)
SELECT o.order_id FROM orders o
WHERE o.amount > (SELECT AVG(amount) FROM orders o2 WHERE o2.customer_id=o.customer_id);

-- FAST: single pass with a window
WITH w AS (SELECT order_id, amount, AVG(amount) OVER (PARTITION BY customer_id) avg_amt FROM orders)
SELECT order_id FROM w WHERE amount > avg_amt;
```
**Why:** the correlated subquery executes per outer row; the window computes the per-customer average in one scan. **Twin:** rewrite "salary > dept average" the same way.

### 157. NOT IN (NULL-unsafe, slow) → NOT EXISTS / anti-join
```sql
-- BUG + SLOW: if the subquery yields any NULL, NOT IN returns no rows
SELECT customer_name FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders);

-- FAST + CORRECT:
SELECT c.customer_name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id);
```
**Why:** `NOT IN` with NULLs evaluates to UNKNOWN and silently drops everything; `NOT EXISTS` is null-safe and the optimizer runs it as an anti-join. **Twin:** products-never-ordered, fixed the same way.

### 158. DISTINCT to dedup a fan-out join → pre-aggregate first
```sql
-- SLOW: join explodes rows, then DISTINCT collapses them (wasteful shuffle)
SELECT DISTINCT c.customer_id, c.customer_name FROM customers c JOIN orders o ON c.customer_id=o.customer_id;

-- FAST: prove existence without the fan-out
SELECT c.customer_id, c.customer_name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id);
```
**Why:** the join multiplies customer rows by their order count before `DISTINCT` removes them; the semi-join short-circuits on the first match. **Twin:** "departments that have employees" without DISTINCT.

### 159. Aggregate-then-join (avoid join fan-out before SUM)
```sql
-- WRONG/SLOW: joining two one-to-many tables double-counts and bloats the scan
SELECT c.customer_id, SUM(o.amount) rev, SUM(r.amount) refunds
FROM customers c JOIN orders o ON c.customer_id=o.customer_id
                 JOIN refunds r ON c.customer_id=r.customer_id GROUP BY c.customer_id;

-- RIGHT/FAST: aggregate each side first, then join the small results
WITH o AS (SELECT customer_id, SUM(amount) rev FROM orders GROUP BY customer_id),
     r AS (SELECT customer_id, SUM(amount) refunds FROM refunds GROUP BY customer_id)
SELECT c.customer_id, o.rev, r.refunds
FROM customers c LEFT JOIN o USING (customer_id) LEFT JOIN r USING (customer_id);
```
**Why:** joining two many-sides creates a cartesian blow-up (orders × refunds per customer) that inflates both the sums and the data scanned. Pre-aggregating keeps each side at one row per key. **Twin:** revenue + review-count per product without double counting.

### 160. Big-data tuning checklist (say these when asked "how would you speed this up?")
```sql
-- Partition pruning: filter the partition column so the engine skips files
SELECT ... FROM events WHERE dt BETWEEN '2026-06-01' AND '2026-06-07';
-- Predicate pushdown + projection: filter early, SELECT only needed columns (never SELECT *)
-- Broadcast the small side of a join:
SELECT /*+ BROADCAST(d) */ ... FROM big_fact f JOIN small_dim d ON f.k=d.k;
-- Skew handling: salt a hot key so one reducer isn't overloaded
SELECT ... FROM (SELECT *, CONCAT(join_key,'_',CAST(FLOOR(RAND()*10) AS INT)) salted_key FROM skewed) ...;
-- Pre-aggregate before shuffle; compact small files; use columnar formats (Parquet/ORC) + appropriate sort/cluster keys.
```
**Say out loud, in order:** (1) prune partitions, (2) push predicates / drop columns, (3) broadcast small dims, (4) pre-aggregate before joins, (5) fix skew via salting, (6) columnar format + compaction. That sequence *is* the senior answer.
**Twin:** Given a 3 TB daily fact joined to a 5 MB dimension and a slow nightly job — list, in order, the four changes you'd make and why. (Answer: broadcast the dim; partition the fact by date and prune; select only needed columns in Parquet; pre-aggregate to the report grain before the final join.)

---

# TIER 15 — Composite "boss level" (multi-table + ranking + LAG/LEAD → final answer)
This is what strict SQL screens actually test now: one realistic business question that forces you to **decompose into chained CTEs** — join several tables → aggregate to a grain → rank within groups → look across rows with LAG/LEAD → filter to the answer. No single trick solves these; the skill is the decomposition.

**Two rules these problems teach:**
1. **One transformation per CTE.** Joins+aggregate in CTE 1, ranking in CTE 2, cross-row comparison in CTE 3, final filter in the SELECT. State the grain at each step.
2. **You cannot nest a window function inside another window function.** To "LAG a rank," you must compute the rank in one CTE and LAG it in the next. (#164 below is built around this gotcha — it's a common rejection trap.)

### 161. Top spender per category + how far ahead of the runner-up
*"For each product category, who is the single biggest-spending customer, and by how much do they beat the #2 spender?"*
```sql
WITH spend AS (                                   -- 3-table join → grain: (category, customer)
  SELECT p.category, c.customer_id, c.customer_name, SUM(o.amount) AS total
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY p.category, c.customer_id, c.customer_name),
ranked AS (                                       -- rank within category + peek at #2 via LEAD
  SELECT category, customer_name, total,
         ROW_NUMBER() OVER (PARTITION BY category ORDER BY total DESC) AS rn,
         LEAD(total)  OVER (PARTITION BY category ORDER BY total DESC) AS runner_up_total
  FROM spend)
SELECT category, customer_name, total,
       total - COALESCE(runner_up_total, 0) AS lead_over_runner_up
FROM ranked
WHERE rn = 1;
```
**Twin:** For each department, the top earner and the gap to the second-highest salary.
```sql
WITH r AS (
  SELECT d.dept_name, e.emp_name, e.salary,
         ROW_NUMBER() OVER (PARTITION BY d.dept_id ORDER BY e.salary DESC) rn,
         LEAD(e.salary) OVER (PARTITION BY d.dept_id ORDER BY e.salary DESC) next_sal
  FROM employees e JOIN departments d ON e.dept_id=d.dept_id)
SELECT dept_name, emp_name, salary, salary - COALESCE(next_sal,0) AS lead_gap
FROM r WHERE rn=1;
```

### 162. Month-over-month decline by country and category
*"Find every (country, category, month) whose revenue fell versus the prior month, with the % drop."*
```sql
WITH monthly AS (                                 -- 3 tables → grain: (country, category, month)
  SELECT c.country, p.category, DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY c.country, p.category, DATE_TRUNC('month', o.order_date)),
growth AS (                                       -- previous month via LAG, partitioned per series
  SELECT country, category, mth, rev,
         LAG(rev) OVER (PARTITION BY country, category ORDER BY mth) AS prev_rev
  FROM monthly)
SELECT country, category, mth, rev, prev_rev,
       (rev - prev_rev) * 100.0 / NULLIF(prev_rev, 0) AS mom_pct
FROM growth
WHERE prev_rev IS NOT NULL AND rev < prev_rev
ORDER BY mom_pct;
```
**Twin:** Creators whose monthly view count dropped vs. the prior month (video_events + creators).
```sql
WITH m AS (SELECT cr.creator_name, DATE_TRUNC('month', v.event_time) mth, COUNT(*) views
           FROM video_events v JOIN creators cr ON v.creator_id=cr.creator_id
           WHERE v.event_type='view' GROUP BY cr.creator_name, DATE_TRUNC('month', v.event_time)),
g AS (SELECT creator_name, mth, views, LAG(views) OVER (PARTITION BY creator_name ORDER BY mth) prev FROM m)
SELECT creator_name, mth, views, prev FROM g WHERE prev IS NOT NULL AND views < prev;
```

### 163. Top-3 customers per country with year-over-year rank movement
*"Rank customers by spend within each country for 2026. For the top 3, show their 2025 rank and whether they moved up, down, stayed, or are new."*
```sql
WITH yearly AS (                                  -- 2 tables → grain: (country, customer, year)
  SELECT c.country, c.customer_id, c.customer_name, YEAR(o.order_date) AS yr, SUM(o.amount) AS total
  FROM orders o JOIN customers c ON o.customer_id = c.customer_id
  GROUP BY c.country, c.customer_id, c.customer_name, YEAR(o.order_date)),
ranked AS (                                       -- rank within (country, year)
  SELECT country, customer_id, customer_name, yr, total,
         RANK() OVER (PARTITION BY country, yr ORDER BY total DESC) AS rnk
  FROM yearly)
SELECT cur.country, cur.customer_name, cur.total,
       cur.rnk AS rank_2026, prev.rnk AS rank_2025,
       CASE WHEN prev.rnk IS NULL      THEN 'new'
            WHEN cur.rnk < prev.rnk    THEN 'up'
            WHEN cur.rnk > prev.rnk    THEN 'down'
            ELSE 'same' END AS movement
FROM ranked cur
LEFT JOIN ranked prev                             -- self-join the ranked CTE across years
  ON cur.country = prev.country AND cur.customer_id = prev.customer_id AND prev.yr = cur.yr - 1
WHERE cur.yr = 2026 AND cur.rnk <= 3
ORDER BY cur.country, cur.rnk;
```
**Twin:** Top-3 products per category this month with last month's rank and movement.
```sql
WITH m AS (SELECT p.category, p.product_name, DATE_TRUNC('month',o.order_date) mth, SUM(o.amount) rev
           FROM orders o JOIN products p ON o.product_id=p.product_id
           GROUP BY p.category, p.product_name, DATE_TRUNC('month',o.order_date)),
r AS (SELECT category, product_name, mth, rev, RANK() OVER (PARTITION BY category, mth ORDER BY rev DESC) rk FROM m)
SELECT cur.category, cur.product_name, cur.rk AS rank_now, prev.rk AS rank_prev
FROM r cur LEFT JOIN r prev
  ON cur.category=prev.category AND cur.product_name=prev.product_name
 AND prev.mth = cur.mth - INTERVAL '1' MONTH
WHERE cur.rk <= 3;
```

### 164. Products whose rank within their category DROPPED month-over-month
*"Which products lost rank inside their category from one month to the next?"* — the **can't-nest-windows** problem.
```sql
WITH m AS (                                       -- grain: (category, product, month)
  SELECT p.category, p.product_id, p.product_name,
         DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o JOIN products p ON o.product_id = p.product_id
  GROUP BY p.category, p.product_id, p.product_name, DATE_TRUNC('month', o.order_date)),
ranked AS (                                       -- STEP 1: compute the rank
  SELECT category, product_id, product_name, mth, rev,
         RANK() OVER (PARTITION BY category, mth ORDER BY rev DESC) AS rnk
  FROM m),
with_prev AS (                                    -- STEP 2: LAG the rank (separate CTE — you can't nest it)
  SELECT category, product_name, mth, rnk,
         LAG(rnk) OVER (PARTITION BY category, product_id ORDER BY mth) AS prev_rnk
  FROM ranked)
SELECT category, product_name, mth, prev_rnk AS rank_last_month, rnk AS rank_this_month
FROM with_prev
WHERE prev_rnk IS NOT NULL AND rnk > prev_rnk      -- bigger number = worse rank = dropped
ORDER BY category, mth;
```
**Twin:** Employees whose salary rank within their department slipped after the latest raise cycle (two snapshots in `salary_history(emp_id, dept_id, cycle, salary)`).
```sql
WITH r AS (SELECT emp_id, dept_id, cycle, salary,
             RANK() OVER (PARTITION BY dept_id, cycle ORDER BY salary DESC) rnk FROM salary_history),
p AS (SELECT emp_id, dept_id, cycle, rnk, LAG(rnk) OVER (PARTITION BY emp_id ORDER BY cycle) prev FROM r)
SELECT emp_id, dept_id, cycle, prev AS rank_before, rnk AS rank_after
FROM p WHERE prev IS NOT NULL AND rnk > prev;
```

### 165. Each customer's 3rd order, with product and days since their 2nd order
*"For customers with at least 3 orders, return the 3rd order's product and how many days passed since their 2nd order."*
```sql
WITH ord AS (                                     -- 3 tables, sequence + previous date together
  SELECT o.customer_id, c.customer_name, o.order_date, p.product_name,
         ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date)      AS seq,
         LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_date
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id)
SELECT customer_name, order_date AS third_order_date, product_name,
       DATEDIFF(order_date, prev_date) AS days_since_second_order
FROM ord
WHERE seq = 3;
```
**Twin:** Each user's 2nd session, with the gap in hours since their 1st (sessions table).
```sql
WITH s AS (SELECT user_id, login_time,
             ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_time) seq,
             LAG(login_time) OVER (PARTITION BY user_id ORDER BY login_time) prev FROM sessions)
SELECT user_id, login_time AS second_session,
       (UNIX_TIMESTAMP(login_time)-UNIX_TIMESTAMP(prev))/3600.0 AS hours_since_first
FROM s WHERE seq = 2;
```

### 166. The month each category's cumulative revenue first crossed $1M
*"For each category, in which month did running revenue first hit one million?"*
```sql
WITH m AS (                                       -- grain: (category, month)
  SELECT p.category, DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o JOIN products p ON o.product_id = p.product_id
  GROUP BY p.category, DATE_TRUNC('month', o.order_date)),
cum AS (                                          -- running total per category
  SELECT category, mth, rev,
         SUM(rev) OVER (PARTITION BY category ORDER BY mth ROWS UNBOUNDED PRECEDING) AS running
  FROM m),
crossed AS (                                      -- first month at/over the threshold
  SELECT category, mth, running,
         ROW_NUMBER() OVER (PARTITION BY category ORDER BY mth) AS rn
  FROM cum WHERE running >= 1000000)
SELECT category, mth AS month_crossed_1M, running
FROM crossed WHERE rn = 1;
```
**Twin:** The day each creator's cumulative views first crossed 100k.
```sql
WITH d AS (SELECT creator_id, CAST(event_time AS DATE) dt, COUNT(*) v FROM video_events WHERE event_type='view' GROUP BY creator_id, CAST(event_time AS DATE)),
c AS (SELECT creator_id, dt, SUM(v) OVER (PARTITION BY creator_id ORDER BY dt ROWS UNBOUNDED PRECEDING) run FROM d),
x AS (SELECT creator_id, dt, run, ROW_NUMBER() OVER (PARTITION BY creator_id ORDER BY dt) rn FROM c WHERE run >= 100000)
SELECT creator_id, dt AS day_crossed_100k FROM x WHERE rn=1;
```

### 167. Category-switch detection between consecutive orders
*"For each customer, list orders where they switched to a different product category than their previous order, with the day gap."*
```sql
WITH ord AS (                                     -- 3 tables; LAG across the joined category column
  SELECT o.customer_id, c.customer_name, o.order_date, p.category,
         LAG(p.category)   OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_category,
         LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_date
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id)
SELECT customer_name, order_date, category, prev_category,
       DATEDIFF(order_date, prev_date) AS days_since_prev
FROM ord
WHERE prev_category IS NOT NULL AND category <> prev_category
ORDER BY customer_name, order_date;
```
**Twin:** Sessions where the user switched device vs. their previous session (`sessions.device`).
```sql
WITH s AS (SELECT user_id, login_time, device,
             LAG(device) OVER (PARTITION BY user_id ORDER BY login_time) prev_dev FROM sessions)
SELECT user_id, login_time, device, prev_dev
FROM s WHERE prev_dev IS NOT NULL AND device <> prev_dev;
```

### 168. Above-country-average AND diversified buyers
*"Customers who spent more than the average customer in their own country AND bought across 3+ distinct categories."*
```sql
WITH spend AS (                                   -- 3 tables → per-customer total + category breadth
  SELECT c.customer_id, c.customer_name, c.country,
         SUM(o.amount) AS total, COUNT(DISTINCT p.category) AS categories
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY c.customer_id, c.customer_name, c.country),
flagged AS (                                      -- window avg per country (no collapse)
  SELECT *, AVG(total) OVER (PARTITION BY country) AS country_avg
  FROM spend)
SELECT customer_name, country, total, categories
FROM flagged
WHERE total > country_avg AND categories >= 3
ORDER BY country, total DESC;
```
**Twin:** Creators with above-region-average views who posted in 3+ distinct months.
```sql
WITH s AS (SELECT cr.creator_id, cr.creator_name, cr.region, COUNT(*) views,
             COUNT(DISTINCT DATE_TRUNC('month', v.event_time)) active_months
           FROM video_events v JOIN creators cr ON v.creator_id=cr.creator_id
           GROUP BY cr.creator_id, cr.creator_name, cr.region),
f AS (SELECT *, AVG(views) OVER (PARTITION BY region) reg_avg FROM s)
SELECT creator_name, region, views, active_months FROM f WHERE views > reg_avg AND active_months >= 3;
```

### 169. Best month per creator and how far it beat their own average (3-way concepts)
*"For each creator, their single best month for views, and the % that month exceeded their average month."*
```sql
WITH cm AS (                                      -- grain: (creator, month)
  SELECT cr.creator_id, cr.creator_name, cr.region,
         DATE_TRUNC('month', v.event_time) AS mth, COUNT(*) AS views
  FROM video_events v JOIN creators cr ON v.creator_id = cr.creator_id
  WHERE v.event_type = 'view'
  GROUP BY cr.creator_id, cr.creator_name, cr.region, DATE_TRUNC('month', v.event_time)),
stats AS (                                        -- per-creator avg (window) + pick best month (rank)
  SELECT creator_id, creator_name, region, mth, views,
         AVG(views)  OVER (PARTITION BY creator_id)                       AS avg_month,
         ROW_NUMBER() OVER (PARTITION BY creator_id ORDER BY views DESC)  AS rn
  FROM cm)
SELECT creator_name, region, mth AS best_month, views,
       (views - avg_month) * 100.0 / avg_month AS pct_above_own_avg
FROM stats
WHERE rn = 1
ORDER BY pct_above_own_avg DESC;
```
**Twin:** Each customer's biggest single order and how much it exceeded their average order value.
```sql
WITH o AS (SELECT customer_id, order_id, amount,
             AVG(amount) OVER (PARTITION BY customer_id) avg_amt,
             ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) rn FROM orders)
SELECT customer_id, order_id, amount, (amount-avg_amt)*100.0/avg_amt AS pct_above_avg
FROM o WHERE rn=1;
```

### 170. Top creator per region — but only among above-median engagement
*"In each region, the creator with the highest like-rate, restricted to creators whose like-rate beats their region's median."*
```sql
WITH eng AS (                                     -- 2 tables → per-creator like-rate
  SELECT cr.region, cr.creator_id, cr.creator_name,
         SUM(CASE WHEN v.event_type='like' THEN 1 ELSE 0 END) * 1.0
         / NULLIF(SUM(CASE WHEN v.event_type='view' THEN 1 ELSE 0 END), 0) AS like_rate
  FROM video_events v JOIN creators cr ON v.creator_id = cr.creator_id
  GROUP BY cr.region, cr.creator_id, cr.creator_name),
med AS (                                          -- region median (the bar)
  SELECT region, PERCENTILE_APPROX(like_rate, 0.5) AS region_median FROM eng GROUP BY region),
above AS (                                         -- keep only above-median, then rank
  SELECT e.region, e.creator_name, e.like_rate,
         RANK() OVER (PARTITION BY e.region ORDER BY e.like_rate DESC) AS rnk
  FROM eng e JOIN med m ON e.region = m.region
  WHERE e.like_rate > m.region_median)
SELECT region, creator_name, like_rate FROM above WHERE rnk = 1;
```
**Twin:** Top product per category whose price is above the category's median price.
```sql
WITH med AS (SELECT category, PERCENTILE_APPROX(price,0.5) m FROM products GROUP BY category),
a AS (SELECT p.category, p.product_name, p.price,
        RANK() OVER (PARTITION BY p.category ORDER BY p.price DESC) rk
      FROM products p JOIN med ON p.category=med.category WHERE p.price > med.m)
SELECT category, product_name, price FROM a WHERE rk=1;
```

### 171. Monthly category leader and when the crown changed hands
*"For each country and month, the #1 category by revenue, the previous month's #1, and a flag when the leader changed."*
```sql
WITH m AS (                                       -- 3 tables → (country, category, month)
  SELECT c.country, p.category, DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY c.country, p.category, DATE_TRUNC('month', o.order_date)),
ranked AS (                                       -- pick the monthly leader
  SELECT country, category, mth, rev,
         ROW_NUMBER() OVER (PARTITION BY country, mth ORDER BY rev DESC) AS rn
  FROM m),
leaders AS (                                      -- LAG the leader across months
  SELECT country, mth, category AS top_category, rev,
         LAG(category) OVER (PARTITION BY country ORDER BY mth) AS prev_top_category
  FROM ranked WHERE rn = 1)
SELECT country, mth, top_category, rev, prev_top_category,
       CASE WHEN prev_top_category IS NULL              THEN 'first_month'
            WHEN top_category <> prev_top_category       THEN 'LEADER CHANGED'
            ELSE 'same' END AS status
FROM leaders
ORDER BY country, mth;
```
**Twin:** Weekly #1 creator per region and when the top creator changed.
```sql
WITH w AS (SELECT cr.region, cr.creator_name, DATE_TRUNC('week', v.event_time) wk, COUNT(*) views
           FROM video_events v JOIN creators cr ON v.creator_id=cr.creator_id
           GROUP BY cr.region, cr.creator_name, DATE_TRUNC('week', v.event_time)),
r AS (SELECT region, creator_name, wk, ROW_NUMBER() OVER (PARTITION BY region, wk ORDER BY views DESC) rn FROM w),
l AS (SELECT region, wk, creator_name top_c, LAG(creator_name) OVER (PARTITION BY region ORDER BY wk) prev_c FROM r WHERE rn=1)
SELECT region, wk, top_c, prev_c, CASE WHEN top_c<>prev_c THEN 'CHANGED' ELSE 'same' END FROM l;
```

### 172. The full monster: cross-region creator leaderboard with growth, rank, and movement
*"For each region and month, take the top 5 creators by views. For each, show their view count, their share of the region's monthly views, their month-over-month growth %, their rank, and how their rank changed vs. last month."*
```sql
WITH cm AS (                                      -- 2 tables → (region, creator, month)
  SELECT cr.region, cr.creator_id, cr.creator_name,
         DATE_TRUNC('month', v.event_time) AS mth, COUNT(*) AS views
  FROM video_events v JOIN creators cr ON v.creator_id = cr.creator_id
  WHERE v.event_type = 'view'
  GROUP BY cr.region, cr.creator_id, cr.creator_name, DATE_TRUNC('month', v.event_time)),
enriched AS (                                     -- share of region-month + rank + MoM growth, all in one pass
  SELECT region, creator_id, creator_name, mth, views,
         views * 100.0 / SUM(views) OVER (PARTITION BY region, mth)            AS pct_of_region,
         RANK()     OVER (PARTITION BY region, mth ORDER BY views DESC)         AS rnk,
         LAG(views) OVER (PARTITION BY region, creator_id ORDER BY mth)         AS prev_views
  FROM cm),
movement AS (                                     -- LAG the rank (separate CTE: no nesting)
  SELECT region, creator_name, mth, views, pct_of_region, rnk, prev_views,
         LAG(rnk) OVER (PARTITION BY region, creator_id ORDER BY mth)           AS prev_rnk
  FROM enriched)
SELECT region, mth, rnk, creator_name, views,
       ROUND(pct_of_region, 1)                                   AS pct_of_region,
       ROUND((views - prev_views) * 100.0 / NULLIF(prev_views,0), 1) AS mom_growth_pct,
       CASE WHEN prev_rnk IS NULL THEN 'new'
            WHEN rnk < prev_rnk   THEN 'up'
            WHEN rnk > prev_rnk   THEN 'down'
            ELSE 'same' END                                      AS rank_movement
FROM movement
WHERE rnk <= 5
ORDER BY region, mth, rnk;
```
**Twin:** Same shape for retail — top 5 products per category per month with share, MoM growth, rank, and rank movement.
```sql
WITH m AS (SELECT p.category, p.product_id, p.product_name, DATE_TRUNC('month',o.order_date) mth, SUM(o.amount) rev
           FROM orders o JOIN products p ON o.product_id=p.product_id
           GROUP BY p.category, p.product_id, p.product_name, DATE_TRUNC('month',o.order_date)),
e AS (SELECT category, product_id, product_name, mth, rev,
        rev*100.0/SUM(rev) OVER (PARTITION BY category, mth) pct,
        RANK() OVER (PARTITION BY category, mth ORDER BY rev DESC) rnk,
        LAG(rev) OVER (PARTITION BY category, product_id ORDER BY mth) prev_rev FROM m),
mv AS (SELECT *, LAG(rnk) OVER (PARTITION BY category, product_id ORDER BY mth) prev_rnk FROM e)
SELECT category, mth, rnk, product_name, rev, ROUND(pct,1) pct,
       ROUND((rev-prev_rev)*100.0/NULLIF(prev_rev,0),1) mom_pct,
       CASE WHEN prev_rnk IS NULL THEN 'new' WHEN rnk<prev_rnk THEN 'up' WHEN rnk>prev_rnk THEN 'down' ELSE 'same' END move
FROM mv WHERE rnk <= 5 ORDER BY category, mth, rnk;
```

---

### How to practice Tier 15
Don't read the solution first. For each prompt: (1) say the **final grain** out loud ("one row per region-month-creator, top 5"), (2) sketch the **CTE chain** on paper — join/aggregate → rank → cross-row → filter — *before* writing SQL, (3) write it, (4) only then check. If you can reliably produce the CTE skeleton for #172 from the prompt alone, you're past the bar at any company screening on SQL today.

---

# APPENDIX — Worked traces for the boss tier (run against the sample data above)
For each Tier-15 problem: what the result *is* on the sample data, and for the trickiest ones, the intermediate CTE output so you can watch the rows transform. **This is the learning loop:** predict the output yourself first, then check here.

### Trace 161 — Top spender per category + lead over runner-up
Step `spend` (join orders+customers+products, sum per category+customer):
| category | customer | total |
|--|--|--|
| electronics | Acme | 400 |
| electronics | Beta | 450 |
| electronics | Gamma | 150 |
| books | Acme | 75 |
| home | Acme | 40 |

Step `ranked` (per category, ROW_NUMBER desc + LEAD): electronics → Beta rn1 (LEAD=400), Acme rn2, Gamma rn3.
**Final (rn=1):**
| category | customer_name | total | lead_over_runner_up |
|--|--|--|--|
| electronics | Beta | 450 | 50 |
| books | Acme | 75 | 75 |
| home | Acme | 40 | 40 |
*Lesson: LEAD reaches "down" the sorted partition to grab the #2 value on the #1's row.*

### Trace 162 — Month-over-month decline by country+category
`monthly` for US electronics: Jan=600 (order1 400 + order3 200), Mar=250 (order6). LAG gives Mar's prev=600.
**Final (rev < prev_rev):**
| country | category | mth | rev | prev_rev | mom_pct |
|--|--|--|--|--|--|
| US | electronics | 2026-03 | 250 | 600 | -58.3 |
*Note: Feb has no US-electronics row, so LAG's "previous" is January — LAG steps to the previous existing row, not the previous calendar month. On real data with a calendar spine you'd guard against that.*

### Trace 163 — Top-3 customers per country, YoY rank movement
Sample has only 2026 orders, so every 2025 rank is NULL → movement = 'new'. US: Acme 515 (rnk1), Beta 450 (rnk2). IN: Gamma 150 (rnk1).
**Final:**
| country | customer_name | total | rank_2026 | rank_2025 | movement |
|--|--|--|--|--|--|
| US | Acme | 515 | 1 | (null) | new |
| US | Beta | 450 | 2 | (null) | new |
| IN | Gamma | 150 | 1 | (null) | new |
*To see real movement, imagine adding 2025 orders where Beta outspent Acme — then Acme's 2026 rank-1 vs 2025 rank-2 prints 'up'.*

### Trace 164 — Products whose category-rank dropped MoM  (the no-nesting problem)
Step `ranked` (rank products within category+month):
| category | product | mth | rev | rnk |
|--|--|--|--|--|
| electronics | Widget Pro | 2026-01 | 600 | 1 |
| electronics | Gizmo | 2026-02 | 150 | 1 |
| electronics | Gizmo | 2026-03 | 250 | 1 |
| books | Notebook | 2026-02 | 75 | 1 |
| home | Desk Lamp | 2026-03 | 40 | 1 |

Step `with_prev` (LAG the rank per product): Gizmo Feb rnk1 (prev null), Gizmo Mar rnk1 (prev 1) → 1 > 1 is false. No product worsened.
**Final: (empty)** — no rank drop exists in this data.
*Lesson — and the whole point of the problem: you compute `rnk` in `ranked`, then LAG it in `with_prev`. You cannot write `LAG(RANK() OVER(...)) OVER(...)` — nesting window functions is a syntax error and a classic rejection. To see a row, add an electronics product that outsells Gizmo in March so Gizmo falls to rank 2.*

### Trace 165 — Each customer's 3rd order + days since 2nd
Only Acme has ≥3 orders: seq1 order1 (2026-01-10), seq2 order2 (2026-02-05), seq3 order5 (2026-03-01, Desk Lamp). prev_date on seq3 = 2026-02-05.
**Final:**
| customer_name | third_order_date | product_name | days_since_second_order |
|--|--|--|--|
| Acme | 2026-03-01 | Desk Lamp | 24 |
*Lesson: ROW_NUMBER assigns the sequence; LAG(order_date) on the same partition+order gives the prior date in one pass — no self-join needed.*

### Trace 166 — Month cumulative revenue crossed $1M per category
Toy totals never reach $1M, so the real query returns **empty**. Illustrating with a **500** threshold instead: `cum` for electronics → Jan running=600 (already ≥500) → crosses in **2026-01**.
| category | month_crossed_500 | running |
|--|--|--|
| electronics | 2026-01 | 600 |
*Lesson: filter the running total to `>= threshold`, then ROW_NUMBER by month and keep rn=1 to grab the *first* qualifying month.*

### Trace 167 — Category switch between consecutive orders (Acme)
Acme sorted: order1 electronics (prev null) → order2 books (prev electronics, switch) → order5 home (prev books, switch). Beta: electronics → electronics (no switch).
**Final:**
| customer_name | order_date | category | prev_category | days_since_prev |
|--|--|--|--|--|
| Acme | 2026-02-05 | books | electronics | 26 |
| Acme | 2026-03-01 | home | books | 24 |
*Lesson: LAG can pull a column from a *joined* table (`p.category`), not just the base table — the join happens before the window.*

### Trace 168 — Above-country-average AND 3+ categories
Per customer: Acme total 515 / 3 categories; Beta 450 / 1; Gamma 150 / 1. US country_avg = (515+450)/2 = 482.5.
Acme 515 > 482.5 and 3 ≥ 3 → kept. Beta 450 < 482.5 → dropped.
**Final:**
| customer_name | country | total | categories |
|--|--|--|--|
| Acme | US | 515 | 3 |
*Lesson: `AVG(total) OVER (PARTITION BY country)` puts the country average on every row without collapsing it, so you can compare each customer to it in the outer WHERE.*

### Trace 169 — Best month per creator vs own average (views)
NovaK views: Jan 1, Feb 2 → avg 1.5, best Feb. RiyaG: Jan 1. LeoM: Feb 1.
**Final (ordered by pct desc):**
| creator_name | region | best_month | views | pct_above_own_avg |
|--|--|--|--|--|
| NovaK | US | 2026-02 | 2 | 33.3 |
| RiyaG | IN | 2026-01 | 1 | 0.0 |
| LeoM | US | 2026-02 | 1 | 0.0 |
*Lesson: one CTE can hold both a partition-wide AVG window and a ROW_NUMBER ranking window; you filter to rn=1 and use the avg in the SELECT.*

### Trace 170 — Top creator per region, above region-median like-rate
like_rate = likes/views: NovaK 1/3 = 0.33, RiyaG 1/1 = 1.0, LeoM 0/1 = 0. US median of {0.33, 0} = 0.165. IN median = 1.0.
Above median: US → NovaK (0.33 > 0.165). IN → none (RiyaG 1.0 is *equal* to the median, not strictly greater).
**Final:**
| region | creator_name | like_rate |
|--|--|--|
| US | NovaK | 0.333 |
*Lesson: filtering to above-median *before* ranking means the rank is computed only over survivors. And `>` vs `>=` matters — a single-creator region is its own median, so strict `>` excludes it.*

### Trace 171 — Monthly category leader per country + when it changed
US leaders by month: Jan electronics (600), Feb books (75, only US category that month), Mar electronics (250 > home 40). LAG the leader across months.
**Final:**
| country | mth | top_category | rev | prev_top_category | status |
|--|--|--|--|--|--|
| US | 2026-01 | electronics | 600 | (null) | first_month |
| US | 2026-02 | books | 75 | electronics | LEADER CHANGED |
| US | 2026-03 | electronics | 250 | books | LEADER CHANGED |
| IN | 2026-02 | electronics | 150 | (null) | first_month |
*Lesson: rank → keep rn=1 (the leader) → LAG the leader's category across months. You're LAGging a value that itself came from a ranking filter.*

### Trace 172 — The monster: top-5 creators per region-month, share + growth + rank + movement
`cm`: US NovaK Jan 1 / Feb 2; US LeoM Feb 1; IN RiyaG Jan 1. Region-month totals: US-Feb = 3.
**Final:**
| region | mth | rnk | creator_name | views | pct_of_region | mom_growth_pct | rank_movement |
|--|--|--|--|--|--|--|--|
| US | 2026-01 | 1 | NovaK | 1 | 100.0 | (null) | new |
| US | 2026-02 | 1 | NovaK | 2 | 66.7 | 100.0 | same |
| US | 2026-02 | 2 | LeoM | 1 | 33.3 | (null) | new |
| IN | 2026-01 | 1 | RiyaG | 1 | 100.0 | (null) | new |
*Lesson — four windows cooperating: `SUM() OVER (region,month)` for share, `RANK() OVER (region,month ORDER BY views)` for rank, `LAG(views) OVER (region,creator ORDER BY month)` for growth — all computable in one CTE because they're independent. But `LAG(rnk)` needs the rank to already exist, so it goes in the **next** CTE (`movement`). That split is the entire architecture of the query.*

---

## The one habit that makes you able to *devise* these
Every boss query is built by answering four questions in order, each becoming one CTE:
1. **What's the base grain?** → the join + GROUP BY (e.g., one row per region-creator-month).
2. **What do I rank or compare within a group?** → the window functions (RANK, ROW_NUMBER, SUM OVER, AVG OVER).
3. **Do I need to look at another row?** → LAG/LEAD — and if it's "LAG a rank," that's a *separate* CTE because windows can't nest.
4. **What's the final filter?** → rn=1, rnk<=N, "changed", "declined", etc., in the outer SELECT.
Write those four answers as English sentences *before* any SQL. The CTE chain then writes itself.
