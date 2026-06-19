# SQL Practice Workbook — Question · Table · Answer
Every item is self-contained: a **Question**, the **table(s)** it runs on, the **Answer** query — then a **Practice** question with its own table and answer. Cover the answer, write your own, then check.

**Dialect:** ANSI SQL with `-- Hive/Spark:` notes where they differ. Tables shown per question are small slices so you can hand-trace the result.

---

# TIER 1 — Basics: SELECT / WHERE / ORDER / LIMIT / DISTINCT

### Q1. Return every column and row from the employees table.
**Table — employees**
| emp_id | emp_name | dept_id | salary | country |
|--|--|--|--|--|
| 1 | Alice | 3 | 150000 | US |
| 2 | Bob | 3 | 120000 | US |
| 3 | Carol | 1 | 130000 | IN |

**Answer:**
```sql
SELECT * FROM employees;
```
**Practice P1.** Return every column and row from the customers table.
**Table — customers**
| customer_id | customer_name | country |
|--|--|--|
| 101 | Acme | US |
| 102 | Beta | IN |

**Answer:**
```sql
SELECT * FROM customers;
```

---

### Q2. Return only each employee's name and salary.
**Table — employees**
| emp_id | emp_name | salary |
|--|--|--|
| 1 | Alice | 150000 |
| 2 | Bob | 120000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees;
```
**Practice P2.** Return only product name and price.
**Table — products**
| product_id | product_name | price |
|--|--|--|
| 1 | Widget Pro | 200 |
| 3 | Notebook | 15 |

**Answer:**
```sql
SELECT product_name, price FROM products;
```

---

### Q3. Return the names of employees in department 3.
**Table — employees**
| emp_name | dept_id |
|--|--|
| Alice | 3 |
| Bob | 3 |
| Carol | 1 |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE dept_id = 3;
```
**Practice P3.** Return the names of customers from `'US'`.
**Table — customers**
| customer_name | country |
|--|--|
| Acme | US |
| Beta | IN |

**Answer:**
```sql
SELECT customer_name FROM customers WHERE country = 'US';
```

---

### Q4. Return employees earning more than 100000, name and salary only.
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Eve | 70000 |
| Bob | 120000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees WHERE salary > 100000;
```
**Practice P4.** Return products priced under 20.
**Table — products**
| product_name | price |
|--|--|
| Widget Pro | 200 |
| Notebook | 15 |

**Answer:**
```sql
SELECT product_name FROM products WHERE price < 20;
```

---

### Q5. Return employees in department 3 who also earn more than 90000.
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Eve | 2 | 70000 |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE dept_id = 3 AND salary > 90000;
```
**Practice P5.** Return orders that are `'shipped'` OR `'delivered'`.
**Table — orders**
| order_id | status |
|--|--|
| 1 | shipped |
| 3 | delivered |
| 4 | cancelled |

**Answer:**
```sql
SELECT order_id FROM orders WHERE status = 'shipped' OR status = 'delivered';
```

---

### Q6. List employees by salary, highest first.
**Table — employees**
| emp_name | salary |
|--|--|
| Bob | 120000 |
| Alice | 150000 |
| Eve | 70000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees ORDER BY salary DESC;
```
**Practice P6.** List products by price, lowest first.
**Table — products**
| product_name | price |
|--|--|
| Widget Pro | 200 |
| Notebook | 15 |

**Answer:**
```sql
SELECT product_name, price FROM products ORDER BY price ASC;
```

---

### Q7. List employees sorted by department ascending, then salary descending within each department.
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Carol | 1 | 130000 |

**Answer:**
```sql
SELECT emp_name, dept_id, salary FROM employees ORDER BY dept_id ASC, salary DESC;
```
**Practice P7.** List orders by customer ascending, then most recent order date first.
**Table — orders**
| order_id | customer_id | order_date |
|--|--|--|
| 1 | 101 | 2026-01-10 |
| 5 | 101 | 2026-03-01 |
| 3 | 102 | 2026-01-20 |

**Answer:**
```sql
SELECT * FROM orders ORDER BY customer_id ASC, order_date DESC;
```

---

### Q8. Return the 5 highest-paid employees.
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Carol | 130000 |
| Bob | 120000 |
| Dan | 95000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees ORDER BY salary DESC LIMIT 5;
```
**Practice P8.** Return the 10 most recent orders.
**Table — orders**
| order_id | order_date |
|--|--|
| 1 | 2026-01-10 |
| 6 | 2026-03-12 |

**Answer:**
```sql
SELECT * FROM orders ORDER BY order_date DESC LIMIT 10;
```

---

### Q9. Return the distinct list of countries customers come from.
**Table — customers**
| customer_id | country |
|--|--|
| 101 | US |
| 102 | US |
| 103 | IN |

**Answer:**
```sql
SELECT DISTINCT country FROM customers;
```
**Practice P9.** Return the distinct product categories.
**Table — products**
| product_id | category |
|--|--|
| 1 | electronics |
| 2 | electronics |
| 3 | books |

**Answer:**
```sql
SELECT DISTINCT category FROM products;
```

---

### Q10. Return distinct (department, country) combinations of employees.
**Table — employees**
| emp_name | dept_id | country |
|--|--|--|
| Alice | 3 | US |
| Bob | 3 | US |
| Dan | 1 | IN |

**Answer:**
```sql
SELECT DISTINCT dept_id, country FROM employees;
```
**Practice P10.** Return distinct (customer, status) pairs from orders.
**Table — orders**
| customer_id | status |
|--|--|
| 101 | shipped |
| 101 | shipped |
| 102 | delivered |

**Answer:**
```sql
SELECT DISTINCT customer_id, status FROM orders;
```

---

# TIER 2 — Filtering operators: IN / BETWEEN / LIKE / NULL

### Q11. Return employees whose department is 1, 2, or 5.
**Table — employees**
| emp_name | dept_id |
|--|--|
| Carol | 1 |
| Eve | 2 |
| Alice | 3 |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE dept_id IN (1, 2, 5);
```
**Practice P11.** Orders whose status is `'pending'` or `'cancelled'`.
**Table — orders**
| order_id | status |
|--|--|
| 4 | cancelled |
| 1 | shipped |

**Answer:**
```sql
SELECT order_id FROM orders WHERE status IN ('pending', 'cancelled');
```

---

### Q12. Return employees NOT in departments 1 or 2.
**Table — employees** (same columns as Q11)
**Answer:**
```sql
SELECT emp_name FROM employees WHERE dept_id NOT IN (1, 2);
```
**Practice P12.** Products not in categories `'toys'` or `'books'`.
**Table — products**
| product_name | category |
|--|--|
| Widget Pro | electronics |
| Notebook | books |

**Answer:**
```sql
SELECT product_name FROM products WHERE category NOT IN ('toys', 'books');
```

---

### Q13. Return employees earning between 50000 and 80000 (inclusive).
**Table — employees**
| emp_name | salary |
|--|--|
| Eve | 70000 |
| Frank | 60000 |
| Alice | 150000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees WHERE salary BETWEEN 50000 AND 80000;
```
**Practice P13.** Orders placed in Q1 2026.
**Table — orders**
| order_id | order_date |
|--|--|
| 1 | 2026-01-10 |
| 6 | 2026-03-12 |

**Answer:**
```sql
SELECT * FROM orders WHERE order_date BETWEEN '2026-01-01' AND '2026-03-31';
```

---

### Q14. Return employees whose name starts with 'A'.
**Table — employees**
| emp_name |
|--|
| Alice |
| Bob |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE emp_name LIKE 'A%';
```
**Practice P14.** Products whose name ends with `'Pro'`.
**Table — products**
| product_name |
|--|
| Widget Pro |
| Gizmo |

**Answer:**
```sql
SELECT product_name FROM products WHERE product_name LIKE '%Pro';
```

---

### Q15. Return customers whose name contains 'son' anywhere.
**Table — customers**
| customer_name |
|--|
| Jackson |
| Acme |

**Answer:**
```sql
SELECT customer_name FROM customers WHERE customer_name LIKE '%son%';
```
**Practice P15.** Employees whose second letter is 'a'.
**Table — employees**
| emp_name |
|--|
| Carol |
| Dan |
| Bob |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE emp_name LIKE '_a%';
```

---

### Q16. Return employees with no manager (top of the org).
**Table — employees**
| emp_name | manager_id |
|--|--|
| Alice | NULL |
| Bob | 1 |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE manager_id IS NULL;
```
**Practice P16.** Customers with no recorded country.
**Table — customers**
| customer_name | country |
|--|--|
| Acme | US |
| Zeta | NULL |

**Answer:**
```sql
SELECT customer_name FROM customers WHERE country IS NULL;
```

---

### Q17. Return employees who DO have a manager.
**Table — employees** (same as Q16)
**Answer:**
```sql
SELECT emp_name FROM employees WHERE manager_id IS NOT NULL;
```
**Practice P17.** Orders that have a non-null amount.
**Table — orders**
| order_id | amount |
|--|--|
| 1 | 400 |
| 9 | NULL |

**Answer:**
```sql
SELECT order_id FROM orders WHERE amount IS NOT NULL;
```

---

### Q18. Return department-3 employees who earn over 90000 OR are based in the US. (Watch operator precedence.)
**Table — employees**
| emp_name | dept_id | salary | country |
|--|--|--|--|
| Alice | 3 | 150000 | US |
| Bob | 3 | 120000 | US |
| Zoe | 3 | 50000 | IN |

**Answer:**
```sql
SELECT emp_name FROM employees WHERE dept_id = 3 AND (salary > 90000 OR country = 'US');
```
**Practice P18.** Shipped orders that are either large (>500) or recent (on/after 2026-06-01).
**Table — orders**
| order_id | status | amount | order_date |
|--|--|--|--|
| 1 | shipped | 400 | 2026-01-10 |
| 7 | shipped | 600 | 2026-02-01 |
| 8 | shipped | 100 | 2026-06-05 |

**Answer:**
```sql
SELECT order_id FROM orders WHERE status = 'shipped' AND (amount > 500 OR order_date >= '2026-06-01');
```

---

# TIER 3 — Aggregation: COUNT / SUM / AVG, GROUP BY, HAVING

### Q19. How many employees are there?
**Table — employees**
| emp_id |
|--|
| 1 |
| 2 |
| 3 |

**Answer:**
```sql
SELECT COUNT(*) FROM employees;
```
**Practice P19.** How many orders are there?
**Table — orders**
| order_id |
|--|
| 1 |
| 2 |

**Answer:**
```sql
SELECT COUNT(*) FROM orders;
```

---

### Q20. Count all employees and, separately, how many have a manager (NULL-aware).
**Table — employees**
| emp_id | manager_id |
|--|--|
| 1 | NULL |
| 2 | 1 |
| 3 | 1 |

**Answer:**
```sql
SELECT COUNT(*) AS all_rows, COUNT(manager_id) AS with_manager FROM employees;
```
*Note: `COUNT(col)` skips NULLs; `COUNT(*)` counts every row.*
**Practice P20.** Total customers vs. customers with a known country.
**Table — customers**
| customer_id | country |
|--|--|
| 101 | US |
| 105 | NULL |

**Answer:**
```sql
SELECT COUNT(*) AS total, COUNT(country) AS known_country FROM customers;
```

---

### Q21. How many distinct departments employ people?
**Table — employees**
| emp_id | dept_id |
|--|--|
| 1 | 3 |
| 2 | 3 |
| 3 | 1 |

**Answer:**
```sql
SELECT COUNT(DISTINCT dept_id) FROM employees;
```
**Practice P21.** How many distinct customers have placed an order?
**Table — orders**
| order_id | customer_id |
|--|--|
| 1 | 101 |
| 2 | 101 |
| 3 | 102 |

**Answer:**
```sql
SELECT COUNT(DISTINCT customer_id) FROM orders;
```

---

### Q22. Return total, average, min, and max order amount.
**Table — orders**
| order_id | amount |
|--|--|
| 1 | 400 |
| 2 | 75 |
| 3 | 200 |

**Answer:**
```sql
SELECT SUM(amount) AS total, AVG(amount) AS mean, MIN(amount) AS lo, MAX(amount) AS hi FROM orders;
```
**Practice P22.** Salary stats (sum/avg/min/max) over employees.
**Table — employees**
| salary |
|--|
| 150000 |
| 70000 |

**Answer:**
```sql
SELECT SUM(salary), AVG(salary), MIN(salary), MAX(salary) FROM employees;
```

---

### Q23. Count employees per department.
**Table — employees**
| emp_id | dept_id |
|--|--|
| 1 | 3 |
| 2 | 3 |
| 3 | 1 |

**Answer:**
```sql
SELECT dept_id, COUNT(*) AS headcount FROM employees GROUP BY dept_id;
```
**Practice P23.** Count orders per status.
**Table — orders**
| order_id | status |
|--|--|
| 1 | shipped |
| 3 | delivered |
| 4 | cancelled |

**Answer:**
```sql
SELECT status, COUNT(*) FROM orders GROUP BY status;
```

---

### Q24. Average salary per department.
**Table — employees** (emp_id, dept_id, salary as above)
**Answer:**
```sql
SELECT dept_id, AVG(salary) AS avg_salary FROM employees GROUP BY dept_id;
```
**Practice P24.** Total revenue per product.
**Table — orders**
| product_id | amount |
|--|--|
| 1 | 400 |
| 1 | 200 |
| 3 | 75 |

**Answer:**
```sql
SELECT product_id, SUM(amount) AS revenue FROM orders GROUP BY product_id;
```

---

### Q25. Count employees per (department, country).
**Table — employees**
| dept_id | country |
|--|--|
| 3 | US |
| 3 | US |
| 1 | IN |

**Answer:**
```sql
SELECT dept_id, country, COUNT(*) AS cnt FROM employees GROUP BY dept_id, country;
```
**Practice P25.** Revenue per (customer, status).
**Table — orders**
| customer_id | status | amount |
|--|--|--|
| 101 | shipped | 400 |
| 101 | shipped | 75 |
| 102 | delivered | 200 |

**Answer:**
```sql
SELECT customer_id, status, SUM(amount) FROM orders GROUP BY customer_id, status;
```

---

### Q26. Return departments with more than 10 employees.
**Table — employees** (emp_id, dept_id)
**Answer:**
```sql
SELECT dept_id, COUNT(*) AS headcount FROM employees GROUP BY dept_id HAVING COUNT(*) > 10;
```
*Note: `WHERE` filters rows before grouping; `HAVING` filters the groups after.*
**Practice P26.** Customers with more than 5 orders.
**Table — orders**
| customer_id |
|--|
| 101 |
| 101 |
| 102 |

**Answer:**
```sql
SELECT customer_id, COUNT(*) AS orders FROM orders GROUP BY customer_id HAVING COUNT(*) > 5;
```

---

### Q27. Among US employees only, return departments whose average salary exceeds 80000.
**Table — employees**
| dept_id | salary | country |
|--|--|--|
| 3 | 150000 | US |
| 3 | 120000 | US |
| 1 | 95000 | IN |

**Answer:**
```sql
SELECT dept_id, AVG(salary) AS avg_salary FROM employees
WHERE country = 'US' GROUP BY dept_id HAVING AVG(salary) > 80000;
```
**Practice P27.** Among shipped orders only, products with revenue over 10000.
**Table — orders**
| product_id | status | amount |
|--|--|--|
| 1 | shipped | 400 |
| 1 | shipped | 200 |
| 2 | cancelled | 150 |

**Answer:**
```sql
SELECT product_id, SUM(amount) AS rev FROM orders WHERE status = 'shipped'
GROUP BY product_id HAVING SUM(amount) > 10000;
```

---

### Q28. Per department, count high earners (>100000) alongside total headcount.
**Table — employees**
| dept_id | salary |
|--|--|
| 3 | 150000 |
| 3 | 120000 |
| 1 | 95000 |

**Answer:**
```sql
SELECT dept_id,
  SUM(CASE WHEN salary > 100000 THEN 1 ELSE 0 END) AS high_earners,
  COUNT(*) AS total
FROM employees GROUP BY dept_id;
```
**Practice P28.** Per customer, count shipped vs. cancelled orders.
**Table — orders**
| customer_id | status |
|--|--|
| 101 | shipped |
| 103 | cancelled |

**Answer:**
```sql
SELECT customer_id,
  SUM(CASE WHEN status='shipped' THEN 1 ELSE 0 END) AS shipped,
  SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders GROUP BY customer_id;
```

---

### Q29. Per department, the fraction of employees earning over 100000.
**Table — employees** (dept_id, salary as Q28)
**Answer:**
```sql
SELECT dept_id, AVG(CASE WHEN salary > 100000 THEN 1.0 ELSE 0.0 END) AS pct_high
FROM employees GROUP BY dept_id;
```
**Practice P29.** Per product, the cancellation rate.
**Table — orders**
| product_id | status |
|--|--|
| 2 | cancelled |
| 2 | shipped |

**Answer:**
```sql
SELECT product_id, AVG(CASE WHEN status='cancelled' THEN 1.0 ELSE 0.0 END) AS cancel_rate
FROM orders GROUP BY product_id;
```

---

### Q30. Top 10 products by revenue.
**Table — orders**
| product_id | amount |
|--|--|
| 1 | 400 |
| 2 | 250 |
| 3 | 75 |

**Answer:**
```sql
SELECT product_id, SUM(amount) AS rev FROM orders GROUP BY product_id ORDER BY rev DESC LIMIT 10;
```
**Practice P30.** Top 5 departments by headcount.
**Table — employees** (emp_id, dept_id)
**Answer:**
```sql
SELECT dept_id, COUNT(*) AS c FROM employees GROUP BY dept_id ORDER BY c DESC LIMIT 5;
```

---

### Q31. Average order value (amount per order) per customer.
**Table — orders**
| customer_id | amount |
|--|--|
| 101 | 400 |
| 101 | 75 |
| 102 | 200 |

**Answer:**
```sql
SELECT customer_id, SUM(amount) / COUNT(*) AS aov FROM orders GROUP BY customer_id;
```
**Practice P31.** Average unit price overall (total amount ÷ total quantity).
**Table — orders**
| quantity | amount |
|--|--|
| 2 | 400 |
| 5 | 75 |

**Answer:**
```sql
SELECT SUM(amount) / SUM(quantity) AS avg_unit_price FROM orders;
```

---

### Q32. Revenue per month.
**Table — orders**
| order_date | amount |
|--|--|
| 2026-01-10 | 400 |
| 2026-01-20 | 200 |
| 2026-02-05 | 75 |

**Answer:**
```sql
SELECT DATE_TRUNC('month', order_date) AS mth, SUM(amount) AS rev
FROM orders GROUP BY DATE_TRUNC('month', order_date) ORDER BY mth;
-- Hive: GROUP BY SUBSTR(CAST(order_date AS STRING),1,7)
```
**Practice P32.** Daily signup counts.
**Table — customers**
| signup_date |
|--|
| 2026-01-05 |
| 2026-01-05 |
| 2026-02-20 |

**Answer:**
```sql
SELECT signup_date, COUNT(*) FROM customers GROUP BY signup_date ORDER BY signup_date;
```

---

# TIER 4 — Joins

### Q33. Show each employee's name with their department name. (INNER JOIN)
**Table — employees**
| emp_name | dept_id |
|--|--|
| Alice | 3 |
| Carol | 1 |

**Table — departments**
| dept_id | dept_name |
|--|--|
| 1 | Engineering |
| 3 | Data |

**Answer:**
```sql
SELECT e.emp_name, d.dept_name
FROM employees e JOIN departments d ON e.dept_id = d.dept_id;
```
**Practice P33.** Show each order with its customer's name.
**Table — orders**
| order_id | customer_id |
|--|--|
| 1 | 101 |

**Table — customers**
| customer_id | customer_name |
|--|--|
| 101 | Acme |

**Answer:**
```sql
SELECT o.order_id, c.customer_name
FROM orders o JOIN customers c ON o.customer_id = c.customer_id;
```

---

### Q34. List all employees with their department name, keeping employees even if their dept is missing. (LEFT JOIN)
**Table — employees**
| emp_name | dept_id |
|--|--|
| Alice | 3 |
| Ghost | 99 |

**Table — departments**
| dept_id | dept_name |
|--|--|
| 3 | Data |

**Answer:**
```sql
SELECT e.emp_name, d.dept_name
FROM employees e LEFT JOIN departments d ON e.dept_id = d.dept_id;
```
**Practice P34.** All customers and their orders (customers with none still appear).
**Table — customers**
| customer_id | customer_name |
|--|--|
| 101 | Acme |
| 104 | Delta |

**Table — orders**
| order_id | customer_id |
|--|--|
| 1 | 101 |

**Answer:**
```sql
SELECT c.customer_name, o.order_id
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id;
```

---

### Q35. Find customers who have never placed an order. (anti-join)
**Table — customers**
| customer_id | customer_name |
|--|--|
| 101 | Acme |
| 104 | Delta |

**Table — orders**
| order_id | customer_id |
|--|--|
| 1 | 101 |

**Answer:**
```sql
SELECT c.customer_name
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_id IS NULL;
```
**Practice P35.** Departments with no employees.
**Table — departments**
| dept_id | dept_name |
|--|--|
| 1 | Engineering |
| 4 | Support |

**Table — employees**
| emp_id | dept_id |
|--|--|
| 3 | 1 |

**Answer:**
```sql
SELECT d.dept_name
FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id
WHERE e.emp_id IS NULL;
```

---

### Q36. Pair each employee with their manager's name. (self-join)
**Table — employees**
| emp_id | emp_name | manager_id |
|--|--|--|
| 1 | Alice | NULL |
| 2 | Bob | 1 |
| 3 | Carol | 1 |

**Answer:**
```sql
SELECT e.emp_name AS employee, m.emp_name AS manager
FROM employees e LEFT JOIN employees m ON e.manager_id = m.emp_id;
```
**Practice P36.** Pairs of customers from the same country (no self-pairs, no duplicates).
**Table — customers**
| customer_id | customer_name | country |
|--|--|--|
| 101 | Acme | US |
| 102 | Beta | US |

**Answer:**
```sql
SELECT a.customer_name, b.customer_name
FROM customers a JOIN customers b
  ON a.country = b.country AND a.customer_id < b.customer_id;
```

---

### Q37. Show order id, customer name, and product name together. (3-table join)
**Table — orders**
| order_id | customer_id | product_id |
|--|--|--|
| 1 | 101 | 1 |

**Table — customers**
| customer_id | customer_name |
|--|--|
| 101 | Acme |

**Table — products**
| product_id | product_name |
|--|--|
| 1 | Widget Pro |

**Answer:**
```sql
SELECT o.order_id, c.customer_name, p.product_name
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products  p ON o.product_id  = p.product_id;
```
**Practice P37.** Employee name, department name, and location.
**Table — employees** (emp_name, dept_id) · **Table — departments** (dept_id, dept_name, location)
**Answer:**
```sql
SELECT e.emp_name, d.dept_name, d.location
FROM employees e JOIN departments d ON e.dept_id = d.dept_id;
```

---

### Q38. Headcount and average salary per department name. (join then aggregate)
**Table — employees** (emp_id, dept_id, salary) · **Table — departments** (dept_id, dept_name)
**Answer:**
```sql
SELECT d.dept_name, COUNT(*) AS headcount, AVG(e.salary) AS avg_salary
FROM employees e JOIN departments d ON e.dept_id = d.dept_id
GROUP BY d.dept_name;
```
**Practice P38.** Revenue per product category.
**Table — orders** (product_id, amount) · **Table — products** (product_id, category)
**Answer:**
```sql
SELECT p.category, SUM(o.amount) AS rev
FROM orders o JOIN products p ON o.product_id = p.product_id
GROUP BY p.category;
```

---

### Q39. Count orders per customer, showing 0 for customers with none. (LEFT JOIN + COUNT)
**Table — customers** (customer_id, customer_name) · **Table — orders** (order_id, customer_id)
**Answer:**
```sql
SELECT c.customer_name, COUNT(o.order_id) AS order_count
FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_name;
```
*Note: `COUNT(o.order_id)` (not `COUNT(*)`) gives 0 for unmatched customers, because the joined columns are NULL.*
**Practice P39.** Total payroll per department, 0 if empty.
**Table — departments** (dept_id, dept_name) · **Table — employees** (dept_id, salary)
**Answer:**
```sql
SELECT d.dept_name, COALESCE(SUM(e.salary), 0) AS total_pay
FROM departments d LEFT JOIN employees e ON d.dept_id = e.dept_id
GROUP BY d.dept_name;
```

---

### Q40. Find customers who have at least one order over 1000. (semi-join with EXISTS)
**Table — customers** (customer_id, customer_name) · **Table — orders** (customer_id, amount)
**Answer:**
```sql
SELECT c.customer_name FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.amount > 1000);
```
**Practice P40.** Departments that have at least one US employee.
**Table — departments** (dept_id, dept_name) · **Table — employees** (dept_id, country)
**Answer:**
```sql
SELECT d.dept_name FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.dept_id = d.dept_id AND e.country = 'US');
```

---

### Q41. Join each customer to their total spend computed in a subquery (avoids double-counting).
**Table — customers** (customer_id, customer_name) · **Table — orders** (customer_id, amount)
**Answer:**
```sql
SELECT c.customer_name, t.total
FROM customers c
JOIN (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) t
  ON c.customer_id = t.customer_id;
```
**Practice P41.** Join each department to its average salary computed separately.
**Table — departments** (dept_id, dept_name) · **Table — employees** (dept_id, salary)
**Answer:**
```sql
SELECT d.dept_name, a.avg_sal
FROM departments d
JOIN (SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id) a
  ON d.dept_id = a.dept_id;
```

---

# TIER 5 — Subqueries

### Q42. Return employees earning more than the company-wide average salary.
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Eve | 70000 |
| Frank | 60000 |

**Answer:**
```sql
SELECT emp_name, salary FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```
**Practice P42.** Orders larger than the average order amount.
**Table — orders**
| order_id | amount |
|--|--|
| 1 | 400 |
| 2 | 75 |

**Answer:**
```sql
SELECT order_id FROM orders WHERE amount > (SELECT AVG(amount) FROM orders);
```

---

### Q43. Return employees in departments located in Seattle. (subquery with IN)
**Table — employees** (emp_name, dept_id) · **Table — departments** (dept_id, location)
**Answer:**
```sql
SELECT emp_name FROM employees
WHERE dept_id IN (SELECT dept_id FROM departments WHERE location = 'Seattle');
```
**Practice P43.** Orders placed by US customers.
**Table — orders** (order_id, customer_id) · **Table — customers** (customer_id, country)
**Answer:**
```sql
SELECT order_id FROM orders
WHERE customer_id IN (SELECT customer_id FROM customers WHERE country = 'US');
```

---

### Q44. Return employees earning more than their OWN department's average. (correlated subquery)
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Dan | 1 | 95000 |

**Answer:**
```sql
SELECT e.emp_name, e.salary FROM employees e
WHERE e.salary > (SELECT AVG(e2.salary) FROM employees e2 WHERE e2.dept_id = e.dept_id);
```
**Practice P44.** Orders above the average amount for that same customer.
**Table — orders** (order_id, customer_id, amount)
**Answer:**
```sql
SELECT o.order_id FROM orders o
WHERE o.amount > (SELECT AVG(o2.amount) FROM orders o2 WHERE o2.customer_id = o.customer_id);
```

---

### Q45. List employees with how far their salary sits above the company average. (scalar subquery in SELECT)
**Table — employees** (emp_name, salary)
**Answer:**
```sql
SELECT emp_name, salary, salary - (SELECT AVG(salary) FROM employees) AS diff_from_mean
FROM employees;
```
**Practice P45.** Each order's amount minus the overall average.
**Table — orders** (order_id, amount)
**Answer:**
```sql
SELECT order_id, amount - (SELECT AVG(amount) FROM orders) AS delta FROM orders;
```

---

### Q46. Return departments whose highest salary exceeds 120000. (subquery in FROM)
**Table — employees**
| dept_id | salary |
|--|--|
| 3 | 150000 |
| 1 | 95000 |

**Answer:**
```sql
SELECT dept_id, max_sal
FROM (SELECT dept_id, MAX(salary) AS max_sal FROM employees GROUP BY dept_id) t
WHERE max_sal > 120000;
```
**Practice P46.** Customers whose total spend exceeds 5000.
**Table — orders** (customer_id, amount)
**Answer:**
```sql
SELECT customer_id, total
FROM (SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id) s
WHERE total > 5000;
```

---

### Q47. Return products never ordered. (NOT EXISTS — null-safe anti-join)
**Table — products** (product_id, product_name) · **Table — orders** (product_id)
**Answer:**
```sql
SELECT p.product_name FROM products p
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.product_id = p.product_id);
```
*Note: prefer `NOT EXISTS` over `NOT IN` — `NOT IN` returns nothing if the subquery yields any NULL.*
**Practice P47.** Customers who have never ordered.
**Table — customers** (customer_id, customer_name) · **Table — orders** (customer_id)
**Answer:**
```sql
SELECT c.customer_name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```

---

# TIER 6 — CTEs (WITH clauses), including recursive

### Q48. Using a CTE, return employees who earn above their department's average.
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Dan | 1 | 95000 |

**Answer:**
```sql
WITH dept_avg AS (
  SELECT dept_id, AVG(salary) AS avg_sal FROM employees GROUP BY dept_id)
SELECT e.emp_name, e.salary, d.avg_sal
FROM employees e JOIN dept_avg d ON e.dept_id = d.dept_id
WHERE e.salary > d.avg_sal;
```
**Practice P48.** Customers spending above their country's average (CTE).
**Table — customers** (customer_id, country) · **Table — orders** (customer_id, amount)
**Answer:**
```sql
WITH country_avg AS (
  SELECT c.country, AVG(o.amount) AS avg_amt
  FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.country)
SELECT o.order_id FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN country_avg ca ON c.country = ca.country
WHERE o.amount > ca.avg_amt;
```

---

### Q49. Use two CTEs to find high earners (>100000) who work in Seattle departments.
**Table — employees** (emp_name, dept_id, salary) · **Table — departments** (dept_id, location)
**Answer:**
```sql
WITH high_earners AS (SELECT * FROM employees WHERE salary > 100000),
     seattle_depts AS (SELECT dept_id FROM departments WHERE location = 'Seattle')
SELECT h.emp_name FROM high_earners h JOIN seattle_depts s ON h.dept_id = s.dept_id;
```
**Practice P49.** Big spenders (>5000 total) who are also US customers.
**Table — orders** (customer_id, amount) · **Table — customers** (customer_id, country)
**Answer:**
```sql
WITH big AS (SELECT customer_id FROM orders GROUP BY customer_id HAVING SUM(amount) > 5000),
     us AS (SELECT customer_id FROM customers WHERE country = 'US')
SELECT b.customer_id FROM big b JOIN us u ON b.customer_id = u.customer_id;
```

---

### Q50. Use chained CTEs to find the top 3 months by revenue.
**Table — orders**
| order_date | amount |
|--|--|
| 2026-01-10 | 600 |
| 2026-02-05 | 75 |
| 2026-03-12 | 250 |

**Answer:**
```sql
WITH monthly AS (
  SELECT DATE_TRUNC('month', order_date) AS mth, SUM(amount) AS rev
  FROM orders GROUP BY DATE_TRUNC('month', order_date)),
ranked AS (SELECT mth, rev, RANK() OVER (ORDER BY rev DESC) AS rnk FROM monthly)
SELECT * FROM ranked WHERE rnk <= 3;
```
**Practice P50.** Top 3 departments by total pay (chained CTEs).
**Table — employees** (dept_id, salary)
**Answer:**
```sql
WITH pay AS (SELECT dept_id, SUM(salary) AS total FROM employees GROUP BY dept_id),
     r AS (SELECT dept_id, total, RANK() OVER (ORDER BY total DESC) rk FROM pay)
SELECT * FROM r WHERE rk <= 3;
```

---

### Q51. Walk the full management hierarchy with a depth level. (recursive CTE)
**Table — employees**
| emp_id | emp_name | manager_id |
|--|--|--|
| 1 | Alice | NULL |
| 2 | Bob | 1 |
| 4 | Dan | 3 |

**Answer:**
```sql
WITH RECURSIVE chain AS (
  SELECT emp_id, emp_name, manager_id, 1 AS lvl FROM employees WHERE manager_id IS NULL
  UNION ALL
  SELECT e.emp_id, e.emp_name, e.manager_id, c.lvl + 1
  FROM employees e JOIN chain c ON e.manager_id = c.emp_id)
SELECT * FROM chain ORDER BY lvl;
-- Hive/Spark: no recursive CTE — use a numbers/iterative approach.
```
**Practice P51.** Generate the number series 1..10 recursively.
**Table —** (none; generated)
**Answer:**
```sql
WITH RECURSIVE nums AS (SELECT 1 AS n UNION ALL SELECT n+1 FROM nums WHERE n < 10)
SELECT n FROM nums;
```

---

# TIER 7 — CASE, conditional logic, NULL handling

### Q52. Label each employee 'high' (≥120000), 'mid' (≥70000), else 'low'.
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Eve | 70000 |
| Frank | 60000 |

**Answer:**
```sql
SELECT emp_name,
  CASE WHEN salary >= 120000 THEN 'high'
       WHEN salary >= 70000  THEN 'mid'
       ELSE 'low' END AS band
FROM employees;
```
**Practice P52.** Label orders 'large' (>1000), 'medium' (>100), else 'small'.
**Table — orders** (order_id, amount)
**Answer:**
```sql
SELECT order_id,
  CASE WHEN amount > 1000 THEN 'large' WHEN amount > 100 THEN 'medium' ELSE 'small' END AS size_band
FROM orders;
```

---

### Q53. Per department, pivot headcount into US and IN columns.
**Table — employees**
| dept_id | country |
|--|--|
| 3 | US |
| 3 | US |
| 1 | IN |

**Answer:**
```sql
SELECT dept_id,
  SUM(CASE WHEN country='US' THEN 1 ELSE 0 END) AS us_cnt,
  SUM(CASE WHEN country='IN' THEN 1 ELSE 0 END) AS in_cnt
FROM employees GROUP BY dept_id;
```
**Practice P53.** Per product, count orders by status as columns.
**Table — orders** (product_id, status)
**Answer:**
```sql
SELECT product_id,
  SUM(CASE WHEN status='shipped' THEN 1 ELSE 0 END) AS shipped,
  SUM(CASE WHEN status='cancelled' THEN 1 ELSE 0 END) AS cancelled
FROM orders GROUP BY product_id;
```

---

### Q54. Compute unit price (amount ÷ quantity) without dividing by zero.
**Table — orders**
| product_id | quantity | amount |
|--|--|--|
| 1 | 2 | 400 |
| 9 | 0 | 0 |

**Answer:**
```sql
SELECT product_id, SUM(amount) / NULLIF(SUM(quantity), 0) AS unit_price
FROM orders GROUP BY product_id;
```
*Note: `NULLIF(x,0)` turns a 0 denominator into NULL, so the division yields NULL instead of erroring.*
**Practice P54.** Show each employee's manager id, defaulting missing ones to 0.
**Table — employees** (emp_name, manager_id)
**Answer:**
```sql
SELECT emp_name, COALESCE(manager_id, 0) AS mgr FROM employees;
```

---

# TIER 8 — String, Date/Time, and Regex

### Q55. Build a label like "Alice (US)" for each employee.
**Table — employees**
| emp_name | country |
|--|--|
| Alice | US |

**Answer:**
```sql
SELECT CONCAT(emp_name, ' (', country, ')') AS label FROM employees;
-- ANSI: emp_name || ' (' || country || ')'
```
**Practice P55.** Build "product - category" labels.
**Table — products** (product_name, category)
**Answer:**
```sql
SELECT CONCAT(product_name, ' - ', category) AS label FROM products;
```

---

### Q56. Extract each employee's first name (text before the first space).
**Table — employees**
| emp_name |
|--|
| Alice Wong |
| Bob Lee |

**Answer:**
```sql
SELECT SPLIT(emp_name, ' ')[0] AS first_name FROM employees;   -- Spark/Hive
-- Postgres: SPLIT_PART(emp_name, ' ', 1)
```
**Practice P56.** Extract the email domain (text after '@').
**Table — customers** (email)
**Answer:**
```sql
SELECT SPLIT(email, '@')[1] AS domain FROM customers;   -- Spark
-- Postgres: SPLIT_PART(email, '@', 2)
```

---

### Q57. How many days ago was each order placed?
**Table — orders**
| order_id | order_date |
|--|--|
| 1 | 2026-01-10 |

**Answer:**
```sql
SELECT order_id, DATEDIFF(CURRENT_DATE, order_date) AS days_ago FROM orders;
-- Postgres: CURRENT_DATE - order_date
```
**Practice P57.** Tenure in days for each employee.
**Table — employees** (emp_name, hire_date)
**Answer:**
```sql
SELECT emp_name, DATEDIFF(CURRENT_DATE, hire_date) AS tenure_days FROM employees;
```

---

### Q58. Revenue by year and month.
**Table — orders**
| order_date | amount |
|--|--|
| 2026-01-10 | 400 |
| 2026-02-05 | 75 |

**Answer:**
```sql
SELECT YEAR(order_date) AS yr, MONTH(order_date) AS mo, SUM(amount) AS rev
FROM orders GROUP BY YEAR(order_date), MONTH(order_date);
```
**Practice P58.** Signups per year.
**Table — customers** (signup_date)
**Answer:**
```sql
SELECT YEAR(signup_date) AS yr, COUNT(*) FROM customers GROUP BY YEAR(signup_date);
```

---

### Q59. Count phone numbers (format 999-999-9999) inside a free-text column, per employee, top 10.
**Table — calls**
| employee | customer_response |
|--|--|
| Sam | call me 415-555-1212 or 415-555-9999 |
| Lee | no number here |

**Answer:**
```sql
SELECT employee,
  SUM(SIZE(REGEXP_EXTRACT_ALL(customer_response, '[0-9]{3}-[0-9]{3}-[0-9]{4}'))) AS cnt
FROM calls GROUP BY employee ORDER BY cnt DESC LIMIT 10;   -- Spark 3.1+
-- Redshift/Oracle: REGEXP_COUNT(customer_response, '...')
```
**Practice P59.** Count hashtags (`#word`) per caption.
**Table — video_events** (video_id, caption)
**Answer:**
```sql
SELECT video_id, SIZE(REGEXP_EXTRACT_ALL(caption, '#[A-Za-z0-9_]+')) AS tags FROM video_events;
```

---

# TIER 9 — Set operations, deduplication, data quality

### Q60. Combine (deduped) customers who shipped an order with those who spent over 1000.
**Table — orders**
| customer_id | status | amount |
|--|--|--|
| 101 | shipped | 400 |
| 102 | delivered | 1200 |

**Answer:**
```sql
SELECT customer_id FROM orders WHERE status = 'shipped'
UNION
SELECT customer_id FROM orders WHERE amount > 1000;
```
*Note: `UNION` removes duplicates; `UNION ALL` keeps them and is faster when you don't need dedup.*
**Practice P60.** All countries appearing in either customers or employees (deduped).
**Table — customers** (country) · **Table — employees** (country)
**Answer:**
```sql
SELECT country FROM customers UNION SELECT country FROM employees;
```

---

### Q61. Find duplicate orders (same customer, product, and date).
**Table — orders**
| order_id | customer_id | product_id | order_date |
|--|--|--|--|
| 1 | 101 | 1 | 2026-01-10 |
| 9 | 101 | 1 | 2026-01-10 |

**Answer:**
```sql
SELECT customer_id, product_id, order_date, COUNT(*) AS c
FROM orders GROUP BY customer_id, product_id, order_date HAVING COUNT(*) > 1;
```
**Practice P61.** Find duplicate employee names.
**Table — employees** (emp_name)
**Answer:**
```sql
SELECT emp_name, COUNT(*) FROM employees GROUP BY emp_name HAVING COUNT(*) > 1;
```

---

### Q62. Keep only the most recent row per order id (deduplicate).
**Table — orders**
| order_id | order_date | amount |
|--|--|--|
| 1 | 2026-01-10 | 400 |
| 1 | 2026-01-12 | 420 |

**Answer:**
```sql
WITH r AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY order_id ORDER BY order_date DESC) AS rn FROM orders)
SELECT * FROM r WHERE rn = 1;
```
**Practice P62.** Keep the latest event per (user, video).
**Table — video_events** (user_id, video_id, event_time)
**Answer:**
```sql
WITH r AS (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY event_time DESC) rn FROM video_events)
SELECT * FROM r WHERE rn = 1;
```

---

### Q63. Audit orders for nulls, negatives, and freshness (data-quality check).
**Table — orders**
| order_id | amount | order_date |
|--|--|--|
| 1 | 400 | 2026-01-10 |
| 9 | NULL | 2026-02-01 |
| 10 | -5 | 2026-02-02 |

**Answer:**
```sql
SELECT COUNT(*) AS rows,
  SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS null_amount,
  SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS negative_amount,
  MAX(order_date) AS latest_load
FROM orders;
```
**Practice P63.** Find orders whose customer id doesn't exist in customers (orphan foreign keys).
**Table — orders** (order_id, customer_id) · **Table — customers** (customer_id)
**Answer:**
```sql
SELECT o.order_id FROM orders o
LEFT JOIN customers c ON o.customer_id = c.customer_id WHERE c.customer_id IS NULL;
```

---

# TIER 10 — Window functions

### Q64. Number employees within each department, highest salary first.
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Carol | 1 | 130000 |

**Answer:**
```sql
SELECT emp_name, dept_id, salary,
  ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn
FROM employees;
```
*Result: Alice rn1 / Bob rn2 in dept 3; Carol rn1 in dept 1.*
**Practice P64.** Number each customer's orders, newest first.
**Table — orders** (order_id, customer_id, order_date)
**Answer:**
```sql
SELECT order_id, customer_id,
  ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) AS rn FROM orders;
```

---

### Q65. Rank all employees by salary using RANK and DENSE_RANK (see how ties differ).
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Carol | 130000 |
| Bob | 130000 |
| Dan | 95000 |

**Answer:**
```sql
SELECT emp_name, salary,
  RANK()       OVER (ORDER BY salary DESC) AS rnk,
  DENSE_RANK() OVER (ORDER BY salary DESC) AS dense
FROM employees;
```
*Result: Alice rnk1/dense1; Carol & Bob both rnk2/dense2; Dan rnk4 (RANK skips to 4) but dense3.*
**Practice P65.** Rank products by price with both functions.
**Table — products** (product_name, price)
**Answer:**
```sql
SELECT product_name, price,
  RANK() OVER (ORDER BY price DESC) AS r, DENSE_RANK() OVER (ORDER BY price DESC) AS dr FROM products;
```

---

### Q66. Return the top 2 highest-paid employees in each department.
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |
| Zoe | 3 | 90000 |
| Carol | 1 | 130000 |

**Answer:**
```sql
WITH r AS (
  SELECT emp_name, dept_id, salary,
    ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) AS rn FROM employees)
SELECT * FROM r WHERE rn <= 2;
```
*Result: dept 3 → Alice, Bob; dept 1 → Carol.*
**Practice P66.** Top 2 products by revenue within each category.
**Table — orders** (product_id, amount) · **Table — products** (product_id, category)
**Answer:**
```sql
WITH rev AS (
  SELECT p.category, p.product_id, SUM(o.amount) AS r
  FROM orders o JOIN products p ON o.product_id=p.product_id GROUP BY p.category, p.product_id),
ranked AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY category ORDER BY r DESC) rn FROM rev)
SELECT * FROM ranked WHERE rn <= 2;
```

---

### Q67. Show each employee's salary next to their department average (without collapsing rows).
**Table — employees**
| emp_name | dept_id | salary |
|--|--|--|
| Alice | 3 | 150000 |
| Bob | 3 | 120000 |

**Answer:**
```sql
SELECT emp_name, dept_id, salary,
  AVG(salary) OVER (PARTITION BY dept_id) AS dept_avg
FROM employees;
```
*Result: both rows show dept_avg = 135000.*
**Practice P67.** Each order's amount alongside its customer's total spend.
**Table — orders** (order_id, customer_id, amount)
**Answer:**
```sql
SELECT order_id, customer_id, amount,
  SUM(amount) OVER (PARTITION BY customer_id) AS customer_total FROM orders;
```

---

### Q68. For each month, show revenue and the previous month's revenue. (LAG)
**Table — orders**
| order_date | amount |
|--|--|
| 2026-01-10 | 600 |
| 2026-02-05 | 75 |
| 2026-03-12 | 250 |

**Answer:**
```sql
WITH m AS (SELECT DATE_TRUNC('month',order_date) mth, SUM(amount) rev
           FROM orders GROUP BY DATE_TRUNC('month',order_date))
SELECT mth, rev, LAG(rev) OVER (ORDER BY mth) AS prev_rev FROM m;
```
*Result: Jan 600/prev NULL; Feb 75/prev 600; Mar 250/prev 75.*
**Practice P68.** Day-over-day session counts with the previous day's count.
**Table — sessions** (login_time)
**Answer:**
```sql
WITH d AS (SELECT CAST(login_time AS DATE) dt, COUNT(*) c FROM sessions GROUP BY CAST(login_time AS DATE))
SELECT dt, c, LAG(c) OVER (ORDER BY dt) AS prev FROM d;
```

---

### Q69. For each event, show the user's NEXT event time. (LEAD)
**Table — video_events**
| user_id | event_time |
|--|--|
| 900 | 2026-01-03 08:00 |
| 900 | 2026-02-04 09:00 |

**Answer:**
```sql
SELECT user_id, event_time,
  LEAD(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS next_event FROM video_events;
```
**Practice P69.** Each order's next order date for the same customer.
**Table — orders** (order_id, customer_id, order_date)
**Answer:**
```sql
SELECT order_id, customer_id, order_date,
  LEAD(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) AS next_order FROM orders;
```

---

### Q70. Month-over-month revenue growth %.
**Table — orders** (order_date, amount as Q68)
**Answer:**
```sql
WITH m AS (SELECT DATE_TRUNC('month',order_date) mth, SUM(amount) rev
           FROM orders GROUP BY DATE_TRUNC('month',order_date))
SELECT mth, rev,
  (rev - LAG(rev) OVER (ORDER BY mth)) * 100.0 / LAG(rev) OVER (ORDER BY mth) AS mom_pct
FROM m;
```
*Result: Feb = (75-600)/600 = -87.5%; Mar = (250-75)/75 = +233%.*
**Practice P70.** MoM growth in monthly active users.
**Table — video_events** (user_id, event_time)
**Answer:**
```sql
WITH m AS (SELECT DATE_TRUNC('month',event_time) mth, COUNT(DISTINCT user_id) u
           FROM video_events GROUP BY DATE_TRUNC('month',event_time))
SELECT mth, u, (u-LAG(u) OVER (ORDER BY mth))*100.0/LAG(u) OVER (ORDER BY mth) AS pct FROM m;
```

---

### Q71. Running (cumulative) revenue by date.
**Table — orders**
| order_date | amount |
|--|--|
| 2026-01-10 | 400 |
| 2026-01-20 | 200 |
| 2026-02-05 | 75 |

**Answer:**
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
  SUM(rev) OVER (ORDER BY order_date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total
FROM d;
```
*Result: 400 → 600 → 675.*
**Practice P71.** Cumulative headcount by hire date.
**Table — employees** (hire_date)
**Answer:**
```sql
WITH h AS (SELECT hire_date, COUNT(*) c FROM employees GROUP BY hire_date)
SELECT hire_date, SUM(c) OVER (ORDER BY hire_date) AS cumulative FROM h;
```

---

### Q72. 7-day moving average of daily revenue.
**Table — orders** (order_date, amount)
**Answer:**
```sql
WITH d AS (SELECT order_date, SUM(amount) rev FROM orders GROUP BY order_date)
SELECT order_date, rev,
  AVG(rev) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7
FROM d;
```
**Practice P72.** 3-day moving average of daily active users.
**Table — video_events** (user_id, event_time)
**Answer:**
```sql
WITH d AS (SELECT CAST(event_time AS DATE) dt, COUNT(DISTINCT user_id) u FROM video_events GROUP BY CAST(event_time AS DATE))
SELECT dt, u, AVG(u) OVER (ORDER BY dt ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS ma3 FROM d;
```

---

### Q73. Split employees into salary quartiles. (NTILE)
**Table — employees**
| emp_name | salary |
|--|--|
| Alice | 150000 |
| Carol | 130000 |
| Bob | 120000 |
| Dan | 95000 |

**Answer:**
```sql
SELECT emp_name, salary, NTILE(4) OVER (ORDER BY salary) AS quartile FROM employees;
```
**Practice P73.** Split customers into 10 deciles by total spend.
**Table — orders** (customer_id, amount)
**Answer:**
```sql
WITH s AS (SELECT customer_id, SUM(amount) t FROM orders GROUP BY customer_id)
SELECT customer_id, t, NTILE(10) OVER (ORDER BY t) AS decile FROM s;
```

---

### Q74. Find the 3rd-highest distinct salary. (DENSE_RANK)
**Table — employees**
| salary |
|--|
| 150000 |
| 130000 |
| 130000 |
| 95000 |

**Answer:**
```sql
WITH r AS (SELECT DISTINCT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS dr FROM employees)
SELECT salary FROM r WHERE dr = 3;
```
*Result: 95000 (150000=1, 130000=2, 95000=3).*
**Practice P74.** 2nd most expensive product per category.
**Table — products** (category, product_name, price)
**Answer:**
```sql
WITH r AS (SELECT category, product_name, price,
             DENSE_RANK() OVER (PARTITION BY category ORDER BY price DESC) dr FROM products)
SELECT * FROM r WHERE dr = 2;
```

---

### Q75. Median salary per department (no percentile function — dual ROW_NUMBER).
**Table — employees**
| dept_id | salary |
|--|--|
| 3 | 150000 |
| 3 | 120000 |
| 3 | 90000 |

**Answer:**
```sql
WITH r AS (
  SELECT dept_id, salary,
    ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary) AS rn,
    COUNT(*)     OVER (PARTITION BY dept_id) AS n
  FROM employees)
SELECT dept_id, AVG(salary) AS median FROM r WHERE rn IN ((n+1)/2, (n+2)/2) GROUP BY dept_id;
```
*Result: dept 3 median = 120000 (middle of 3 rows).*
**Practice P75.** Median order amount per status.
**Table — orders** (status, amount)
**Answer:**
```sql
WITH r AS (SELECT status, amount,
             ROW_NUMBER() OVER (PARTITION BY status ORDER BY amount) rn,
             COUNT(*) OVER (PARTITION BY status) n FROM orders)
SELECT status, AVG(amount) AS median FROM r WHERE rn IN ((n+1)/2,(n+2)/2) GROUP BY status;
```

---

# TIER 11 — Advanced analytical patterns

### Q76. Find users active 3+ consecutive days. (gaps & islands)
**Table — video_events**
| user_id | event_time |
|--|--|
| 900 | 2026-01-01 |
| 900 | 2026-01-02 |
| 900 | 2026-01-03 |
| 901 | 2026-01-01 |

**Answer:**
```sql
WITH a AS (SELECT DISTINCT user_id, CAST(event_time AS DATE) d FROM video_events),
g AS (SELECT user_id, d,
        DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY d)) AS grp FROM a)
SELECT user_id, MIN(d) AS streak_start, COUNT(*) AS streak_len
FROM g GROUP BY user_id, grp HAVING COUNT(*) >= 3;
```
*Trick: for consecutive dates, `date − row_number` is constant, forming a group key. Result: user 900, 3-day streak.*
**Practice P76.** Customers ordering 3+ days in a row.
**Table — orders** (customer_id, order_date)
**Answer:**
```sql
WITH a AS (SELECT DISTINCT customer_id, order_date d FROM orders),
g AS (SELECT customer_id, d, DATE_SUB(d, ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY d)) grp FROM a)
SELECT customer_id, MIN(d), COUNT(*) FROM g GROUP BY customer_id, grp HAVING COUNT(*)>=3;
```

---

### Q77. Split each user's events into sessions (a >30-min gap starts a new session).
**Table — video_events**
| user_id | event_time |
|--|--|
| 900 | 2026-01-01 08:00 |
| 900 | 2026-01-01 08:10 |
| 900 | 2026-01-01 12:00 |

**Answer:**
```sql
WITH e AS (
  SELECT user_id, event_time,
    LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_t FROM video_events),
flagged AS (
  SELECT user_id, event_time,
    CASE WHEN prev_t IS NULL OR (UNIX_TIMESTAMP(event_time)-UNIX_TIMESTAMP(prev_t)) > 1800
         THEN 1 ELSE 0 END AS new_session FROM e)
SELECT user_id, event_time,
  SUM(new_session) OVER (PARTITION BY user_id ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS session_id
FROM flagged;
```
*Result: first two events session 1; the 12:00 event (3h59m gap) starts session 2.*
**Practice P77.** Sessionize orders where a >7-day gap starts a new "shopping spell".
**Table — orders** (customer_id, order_date)
**Answer:**
```sql
WITH e AS (SELECT customer_id, order_date,
             LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date) prev FROM orders),
f AS (SELECT customer_id, order_date,
        CASE WHEN prev IS NULL OR DATEDIFF(order_date, prev) > 7 THEN 1 ELSE 0 END ns FROM e)
SELECT customer_id, order_date,
  SUM(ns) OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS spell FROM f;
```

---

### Q78. Find the moment with the most users online at once. (sweep line)
**Table — sessions**
| user_id | login_time | logout_time |
|--|--|--|
| 900 | 08:00 | 09:00 |
| 901 | 08:30 | 08:45 |
| 902 | 08:40 | 10:00 |

**Answer:**
```sql
WITH ev AS (
  SELECT login_time AS ts, 1 AS delta FROM sessions
  UNION ALL SELECT logout_time, -1 FROM sessions),
run AS (
  SELECT ts, SUM(delta) OVER (ORDER BY ts, delta DESC ROWS UNBOUNDED PRECEDING) AS concurrent FROM ev)
SELECT ts, concurrent FROM run ORDER BY concurrent DESC LIMIT 1;
```
*Trick: +1 per login, −1 per logout, sort by time, running sum → the max is peak concurrency. Result: 08:40 with 3 online.*
**Practice P78.** Peak simultaneous active video streams.
**Table — streams** (start_ts, end_ts)
**Answer:**
```sql
WITH ev AS (SELECT start_ts ts,1 d FROM streams UNION ALL SELECT end_ts,-1 FROM streams),
run AS (SELECT ts, SUM(d) OVER (ORDER BY ts, d DESC ROWS UNBOUNDED PRECEDING) c FROM ev)
SELECT ts, c FROM run ORDER BY c DESC LIMIT 1;
```

---

### Q79. Day-1 retention rate: fraction of users who return the day after their first activity.
**Table — video_events**
| user_id | event_time |
|--|--|
| 900 | 2026-01-01 |
| 900 | 2026-01-02 |
| 901 | 2026-01-01 |

**Answer:**
```sql
WITH first_day AS (SELECT user_id, MIN(CAST(event_time AS DATE)) d0 FROM video_events GROUP BY user_id),
returned AS (
  SELECT f.user_id FROM first_day f
  JOIN video_events v ON v.user_id=f.user_id AND CAST(v.event_time AS DATE)=DATE_ADD(f.d0,1))
SELECT COUNT(DISTINCT r.user_id)*1.0 / COUNT(DISTINCT f.user_id) AS d1_retention
FROM first_day f LEFT JOIN returned r ON f.user_id=r.user_id;
```
*Result: 900 returned on day 2, 901 didn't → 0.5.*
**Practice P79.** Day-1 retention for customers (first order date + 1).
**Table — orders** (customer_id, order_date)
**Answer:**
```sql
WITH f AS (SELECT customer_id, MIN(order_date) d0 FROM orders GROUP BY customer_id),
ret AS (SELECT f.customer_id FROM f JOIN orders o ON o.customer_id=f.customer_id AND o.order_date=DATE_ADD(f.d0,1))
SELECT COUNT(DISTINCT ret.customer_id)*1.0/COUNT(DISTINCT f.customer_id)
FROM f LEFT JOIN ret ON f.customer_id=ret.customer_id;
```

---

### Q80. Engagement funnel: count distinct users at view, like, and purchase stages.
**Table — video_events**
| user_id | event_type |
|--|--|
| 900 | view |
| 900 | purchase |
| 901 | view |

**Answer:**
```sql
SELECT
  COUNT(DISTINCT CASE WHEN event_type='view'     THEN user_id END) AS viewers,
  COUNT(DISTINCT CASE WHEN event_type='like'     THEN user_id END) AS likers,
  COUNT(DISTINCT CASE WHEN event_type='purchase' THEN user_id END) AS buyers
FROM video_events;
```
*Result: viewers 2, likers 0, buyers 1.*
**Practice P80.** Order funnel: carted vs shipped vs delivered distinct customers.
**Table — orders** (customer_id, status)
**Answer:**
```sql
SELECT
  COUNT(DISTINCT CASE WHEN status='carted'    THEN customer_id END) AS carted,
  COUNT(DISTINCT CASE WHEN status='shipped'   THEN customer_id END) AS shipped,
  COUNT(DISTINCT CASE WHEN status='delivered' THEN customer_id END) AS delivered
FROM orders;
```

---

### Q81. Pivot each customer's revenue into 2025 and 2026 columns.
**Table — orders**
| customer_id | order_date | amount |
|--|--|--|
| 101 | 2025-12-01 | 100 |
| 101 | 2026-01-10 | 400 |

**Answer:**
```sql
SELECT customer_id,
  SUM(CASE WHEN YEAR(order_date)=2025 THEN amount ELSE 0 END) AS rev_2025,
  SUM(CASE WHEN YEAR(order_date)=2026 THEN amount ELSE 0 END) AS rev_2026
FROM orders GROUP BY customer_id;
```
*Result: customer 101 → 100, 400.*
**Practice P81.** Pivot event counts by type per creator.
**Table — video_events** (creator_id, event_type)
**Answer:**
```sql
SELECT creator_id,
  SUM(CASE WHEN event_type='view' THEN 1 ELSE 0 END) AS views,
  SUM(CASE WHEN event_type='like' THEN 1 ELSE 0 END) AS likes
FROM video_events GROUP BY creator_id;
```

---

# TIER 12 — Data-engineering / Hive-Spark specific

### Q82. Incrementally load only orders newer than what's already in the warehouse. (high-water mark)
**Table — orders** (source) and **Table — dw_orders** (target), both with order_date.
**Answer:**
```sql
INSERT INTO dw_orders
SELECT * FROM orders
WHERE order_date > (SELECT COALESCE(MAX(order_date), '1970-01-01') FROM dw_orders);
```
*Trick: the target's max date is the watermark; only newer rows load — the basis of incremental pipelines.*
**Practice P82.** Load only source_table rows changed after the target's max updated_at.
**Table — source_table / target_table** (updated_at)
**Answer:**
```sql
INSERT INTO target_table
SELECT * FROM source_table s
WHERE s.updated_at > (SELECT COALESCE(MAX(updated_at), '1970-01-01') FROM target_table);
```

---

### Q83. From a change-data-capture stream, keep the latest non-deleted version per key.
**Table — cdc_stream**
| pk | op_type | op_ts | val |
|--|--|--|--|
| 1 | INSERT | 08:00 | a |
| 1 | UPDATE | 09:00 | b |
| 2 | DELETE | 10:00 | x |

**Answer:**
```sql
WITH r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY pk ORDER BY op_ts DESC) AS rn FROM cdc_stream)
SELECT * FROM r WHERE rn = 1 AND op_type <> 'DELETE';
```
*Result: pk 1 = "b"; pk 2 dropped (latest op is DELETE).*
**Practice P83.** Latest non-deleted creator profile from a change log.
**Table — creator_changes** (creator_id, op, change_ts)
**Answer:**
```sql
WITH r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY creator_id ORDER BY change_ts DESC) rn FROM creator_changes)
SELECT * FROM r WHERE rn=1 AND op <> 'DELETE';
```

---

### Q84. Detect creators whose region changed vs. the current dimension row (SCD Type 2 input).
**Table — staging_creators**
| creator_id | region |
|--|--|
| 1 | EU |

**Table — dim_creator** (is_current = TRUE rows only)
| creator_id | region | is_current |
|--|--|--|
| 1 | US | TRUE |

**Answer:**
```sql
SELECT s.creator_id, d.region AS old_region, s.region AS new_region
FROM staging_creators s
JOIN dim_creator d ON s.creator_id = d.creator_id AND d.is_current
WHERE s.region <> d.region;
```
*These are the rows you'd expire (set effective_to) and re-insert as a new current version — SCD Type 2.*
**Practice P84.** Products whose price changed vs. the current dimension row.
**Table — staging_products / dim_product** (product_id, price, is_current)
**Answer:**
```sql
SELECT s.product_id, d.price AS old_price, s.price AS new_price
FROM staging_products s JOIN dim_product d ON s.product_id=d.product_id AND d.is_current
WHERE s.price <> d.price;
```

---

### Q85. Read only one week of a date-partitioned table. (partition pruning)
**Table — events_partitioned** (partition key `dt`)
**Answer:**
```sql
SELECT * FROM events_partitioned WHERE dt BETWEEN '2026-06-01' AND '2026-06-07';
```
*The engine skips every partition outside the range — the single biggest scan-reduction lever.*
**Practice P85.** Read just one day's partition of orders.
**Table — orders_partitioned** (dt)
**Answer:**
```sql
SELECT * FROM orders_partitioned WHERE dt = '2026-06-15';
```

---

### Q86. Explode a comma-separated hashtags column into one row per tag. (Spark/Hive)
**Table — video_events**
| user_id | hashtags |
|--|--|
| 900 | fyp,dance,viral |

**Answer:**
```sql
SELECT user_id, tag
FROM video_events
LATERAL VIEW EXPLODE(SPLIT(hashtags, ',')) t AS tag;
```
*Result: three rows — fyp, dance, viral.*
**Practice P86.** Explode a comma-separated categories column in products.
**Table — products** (product_id, categories)
**Answer:**
```sql
SELECT product_id, cat FROM products LATERAL VIEW EXPLODE(SPLIT(categories, ',')) c AS cat;
```

---

### Q87. Top earner per department using QUALIFY (filter on a window without a subquery).
**Table — employees** (emp_name, dept_id, salary)
**Answer:**
```sql
SELECT emp_name, dept_id, salary
FROM employees
QUALIFY ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY salary DESC) = 1;
-- Hive/Spark (no QUALIFY): wrap in a CTE and filter WHERE rn = 1
```
**Practice P87.** Latest order per customer using QUALIFY.
**Table — orders** (order_id, customer_id, order_date)
**Answer:**
```sql
SELECT * FROM orders
QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1;
```

---

# TIER 13 — God tier (multi-step patterns that gate offers)

### Q88. Running total that RESETS whenever a flag fires.
**Table — orders**
| customer_id | order_date | amount | is_reset |
|--|--|--|--|
| 101 | 2026-01-01 | 100 | 1 |
| 101 | 2026-01-05 | 50 | 0 |
| 101 | 2026-02-01 | 80 | 1 |

**Answer:**
```sql
WITH base AS (
  SELECT customer_id, order_date, amount, is_reset,
    SUM(is_reset) OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS seg
  FROM orders)
SELECT customer_id, order_date, amount,
  SUM(amount) OVER (PARTITION BY customer_id, seg ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS running
FROM base;
```
*Trick: the cumulative count of the reset flag is the segment id. Result: 100 → 150 → 80 (resets at row 3).*
**Practice P88.** Running event count per user that resets each calendar day.
**Table — video_events** (user_id, event_time)
**Answer:**
```sql
WITH b AS (SELECT user_id, event_time, CAST(event_time AS DATE) dt FROM video_events)
SELECT user_id, event_time,
  COUNT(*) OVER (PARTITION BY user_id, dt ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS daily_seq FROM b;
```

---

### Q89. Smallest set of products making up 80% of revenue. (Pareto)
**Table — orders**
| product_id | amount |
|--|--|
| 1 | 600 |
| 2 | 250 |
| 3 | 75 |

**Answer:**
```sql
WITH rev AS (SELECT product_id, SUM(amount) r FROM orders GROUP BY product_id),
c AS (SELECT product_id, r,
        SUM(r) OVER (ORDER BY r DESC ROWS UNBOUNDED PRECEDING) AS running,
        SUM(r) OVER () AS total FROM rev)
SELECT product_id, r, running/total AS cum_share
FROM c WHERE running - r < 0.8 * total;
```
*Trick: keep rows until the running total first crosses 80%. Total=925, 80%=740. Product 1 (600) and product 2 (cum 850) included; product 3 excluded.*
**Practice P89.** Top creators producing 90% of all views.
**Table — video_events** (creator_id) — counts as views
**Answer:**
```sql
WITH v AS (SELECT creator_id, COUNT(*) c FROM video_events GROUP BY creator_id),
cc AS (SELECT creator_id, c, SUM(c) OVER (ORDER BY c DESC ROWS UNBOUNDED PRECEDING) run, SUM(c) OVER () tot FROM v)
SELECT creator_id FROM cc WHERE run - c < 0.9 * tot;
```

---

### Q90. As-of join: for each trade, attach the latest price quoted at or before it.
**Table — trades**
| trade_id | symbol | trade_time |
|--|--|--|
| 1 | ABC | 10:05 |

**Table — quotes**
| symbol | quote_time | price |
|--|--|--|
| ABC | 10:00 | 50 |
| ABC | 10:04 | 52 |
| ABC | 10:10 | 55 |

**Answer:**
```sql
WITH ranked AS (
  SELECT t.trade_id, t.symbol, t.trade_time, q.price, q.quote_time,
    ROW_NUMBER() OVER (PARTITION BY t.trade_id ORDER BY q.quote_time DESC) AS rn
  FROM trades t JOIN quotes q ON q.symbol = t.symbol AND q.quote_time <= t.trade_time)
SELECT trade_id, symbol, trade_time, price FROM ranked WHERE rn = 1;
```
*Trick: join on `quote_time <= trade_time`, then ROW_NUMBER to keep the most recent. Result: price 52 (the 10:10 quote is after the trade, excluded).*
**Practice P90.** For each purchase, the same user's most recent view before it.
**Table — video_events** (event_id, user_id, event_type, event_time)
**Answer:**
```sql
WITH r AS (
  SELECT p.event_id, p.user_id, p.event_time buy_t, v.event_time view_t,
    ROW_NUMBER() OVER (PARTITION BY p.event_id ORDER BY v.event_time DESC) rn
  FROM video_events p
  JOIN video_events v ON v.user_id=p.user_id AND v.event_type='view' AND v.event_time <= p.event_time
  WHERE p.event_type='purchase')
SELECT event_id, user_id, buy_t, view_t FROM r WHERE rn=1;
```

---

### Q91. Merge overlapping booking intervals into consolidated blocks.
**Table — bookings**
| room_id | start_ts | end_ts |
|--|--|--|
| A | 09:00 | 10:00 |
| A | 09:30 | 11:00 |
| A | 12:00 | 13:00 |

**Answer:**
```sql
WITH ordered AS (
  SELECT room_id, start_ts, end_ts,
    MAX(end_ts) OVER (PARTITION BY room_id ORDER BY start_ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_max_end
  FROM bookings),
grp AS (
  SELECT room_id, start_ts, end_ts,
    SUM(CASE WHEN prev_max_end IS NULL OR start_ts > prev_max_end THEN 1 ELSE 0 END)
      OVER (PARTITION BY room_id ORDER BY start_ts ROWS UNBOUNDED PRECEDING) AS g
  FROM ordered)
SELECT room_id, MIN(start_ts) AS merged_start, MAX(end_ts) AS merged_end FROM grp GROUP BY room_id, g;
```
*Trick: a new block starts when start_ts > running max(end_ts). Result: 09:00–11:00 (rows 1+2 merge) and 12:00–13:00.*
**Practice P91.** Merge overlapping/contiguous active subscription windows per user.
**Table — subs** (user_id, start_date, end_date)
**Answer:**
```sql
WITH o AS (SELECT user_id, start_date s, end_date e,
             MAX(end_date) OVER (PARTITION BY user_id ORDER BY start_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pe FROM subs),
g AS (SELECT user_id, s, e, SUM(CASE WHEN pe IS NULL OR s>pe THEN 1 ELSE 0 END) OVER (PARTITION BY user_id ORDER BY s ROWS UNBOUNDED PRECEDING) grp FROM o)
SELECT user_id, MIN(s), MAX(e) FROM g GROUP BY user_id, grp;
```

---

### Q92. Cumulative DISTINCT count of videos watched per user over time.
**Table — video_events**
| user_id | video_id | event_time |
|--|--|--|
| 900 | 11 | 08:00 |
| 900 | 11 | 09:00 |
| 900 | 12 | 10:00 |

**Answer:**
```sql
WITH firsts AS (
  SELECT user_id, video_id, event_time,
    ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY event_time) AS occ FROM video_events)
SELECT user_id, event_time,
  SUM(CASE WHEN occ=1 THEN 1 ELSE 0 END)
    OVER (PARTITION BY user_id ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS distinct_videos_so_far
FROM firsts;
```
*Trick: windows can't COUNT DISTINCT — so count a value only on its FIRST occurrence, then running-sum the flags. Result: 1 → 1 (repeat) → 2.*
**Practice P92.** Cumulative distinct products a customer has ordered over time.
**Table — orders** (customer_id, product_id, order_date)
**Answer:**
```sql
WITH f AS (SELECT customer_id, product_id, order_date,
             ROW_NUMBER() OVER (PARTITION BY customer_id, product_id ORDER BY order_date) occ FROM orders)
SELECT customer_id, order_date,
  SUM(CASE WHEN occ=1 THEN 1 ELSE 0 END) OVER (PARTITION BY customer_id ORDER BY order_date ROWS UNBOUNDED PRECEDING) AS distinct_products FROM f;
```

---

### Q93. Forward-fill the last known status into rows where status is NULL.
**Table — user_status**
| user_id | event_time | status |
|--|--|--|
| 900 | 08:00 | active |
| 900 | 09:00 | NULL |
| 900 | 10:00 | NULL |

**Answer:**
```sql
WITH s AS (
  SELECT user_id, event_time, status,
    COUNT(status) OVER (PARTITION BY user_id ORDER BY event_time ROWS UNBOUNDED PRECEDING) AS grp
  FROM user_status)
SELECT user_id, event_time, MAX(status) OVER (PARTITION BY user_id, grp) AS status_filled FROM s;
-- Where supported: LAST_VALUE(status) IGNORE NULLS OVER (...) does this directly.
```
*Trick: cumulative COUNT of non-nulls forms a group; the one non-null per group fills the rest. Result: active, active, active.*
**Practice P93.** Forward-fill the last known price per product across a daily timeline.
**Table — price_daily** (product_id, dt, price)
**Answer:**
```sql
WITH s AS (SELECT product_id, dt, price,
             COUNT(price) OVER (PARTITION BY product_id ORDER BY dt ROWS UNBOUNDED PRECEDING) g FROM price_daily)
SELECT product_id, dt, MAX(price) OVER (PARTITION BY product_id, g) AS price_filled FROM s;
```

---

### Q94. Find customers active in EVERY month of 2026 (all 12).
**Table — orders**
| customer_id | order_date |
|--|--|
| 101 | 2026-01-10 |
| 101 | 2026-02-05 |

**Answer:**
```sql
SELECT customer_id FROM orders WHERE YEAR(order_date)=2026
GROUP BY customer_id HAVING COUNT(DISTINCT MONTH(order_date)) = 12;
```
*Trick: count distinct months and require it to equal the full coverage (12).*
**Practice P94.** Users active every single day of June 2026 (30 days).
**Table — video_events** (user_id, event_time)
**Answer:**
```sql
SELECT user_id FROM video_events
WHERE event_time >= '2026-06-01' AND event_time < '2026-07-01'
GROUP BY user_id HAVING COUNT(DISTINCT CAST(event_time AS DATE)) = 30;
```

---

# TIER 14 — Optimization & rewrite reasoning (say the *why* out loud)

### Q95. This correlated subquery is slow. Rewrite it to run in one pass.
**Slow form:**
```sql
SELECT o.order_id FROM orders o
WHERE o.amount > (SELECT AVG(amount) FROM orders o2 WHERE o2.customer_id = o.customer_id);
```
**Answer (fast — window function):**
```sql
WITH w AS (SELECT order_id, amount, AVG(amount) OVER (PARTITION BY customer_id) avg_amt FROM orders)
SELECT order_id FROM w WHERE amount > avg_amt;
```
*Why: the correlated subquery re-scans orders once per outer row; the window computes each customer's average in a single scan.*
**Practice P95.** Rewrite "employees earning above their dept average" without a correlated subquery.
**Answer:**
```sql
WITH w AS (SELECT emp_name, salary, AVG(salary) OVER (PARTITION BY dept_id) a FROM employees)
SELECT emp_name FROM w WHERE salary > a;
```

---

### Q96. This NOT IN can silently return nothing and is slow. Fix it.
**Buggy/slow form:**
```sql
SELECT customer_name FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders);
```
**Answer (correct + fast — NOT EXISTS):**
```sql
SELECT c.customer_name FROM customers c
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id);
```
*Why: if the subquery returns any NULL, `NOT IN` evaluates to UNKNOWN and drops every row. `NOT EXISTS` is null-safe and runs as an anti-join.*
**Practice P96.** Rewrite "products never ordered" safely.
**Answer:**
```sql
SELECT p.product_name FROM products p
WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.product_id = p.product_id);
```

---

### Q97. Two one-to-many joins inflate the sums. Fix the double-counting.
**Wrong/slow form:**
```sql
SELECT c.customer_id, SUM(o.amount) rev, SUM(r.amount) refunds
FROM customers c JOIN orders o ON c.customer_id=o.customer_id
                 JOIN refunds r ON c.customer_id=r.customer_id GROUP BY c.customer_id;
```
**Answer (aggregate each side first, then join):**
```sql
WITH o AS (SELECT customer_id, SUM(amount) rev FROM orders GROUP BY customer_id),
     r AS (SELECT customer_id, SUM(amount) refunds FROM refunds GROUP BY customer_id)
SELECT c.customer_id, o.rev, r.refunds
FROM customers c LEFT JOIN o USING(customer_id) LEFT JOIN r USING(customer_id);
```
*Why: joining two many-sides produces orders × refunds rows per customer, inflating both sums. Pre-aggregating keeps one row per key.*
**Practice P97.** A 3 TB fact joined to a 5 MB dim runs slow nightly. Name the four fixes, in order.
**Answer (say this):** 1) **broadcast** the small dim (`/*+ BROADCAST(d) */`) to avoid a shuffle; 2) **partition** the fact by date and **prune** to the loaded range; 3) read **only needed columns** in **Parquet** (projection + predicate pushdown); 4) **pre-aggregate** to the report grain before the final join. If one key is hot, **salt** it to fix skew.

---

# TIER 15 — Boss level (3+ tables + ranking + LAG/LEAD → answer)
Each shows the input tables, the query, and the **expected output** on that data. The skill is decomposition: join/aggregate → rank → look across rows → filter. (Reused dataset: customers 101 Acme/US, 102 Beta/US, 103 Gamma/IN; products 1 Widget Pro/electronics, 2 Gizmo/electronics, 3 Notebook/books, 4 Desk Lamp/home.)

### Q98. For each category, the biggest-spending customer and the gap to the runner-up.
**Table — orders**
| order_id | customer_id | product_id | amount |
|--|--|--|--|
| 1 | 101 | 1 | 400 |
| 2 | 101 | 3 | 75 |
| 3 | 102 | 1 | 200 |
| 5 | 101 | 4 | 40 |
| 6 | 102 | 2 | 250 |

**Tables — customers** (id→name) and **products** (id→category) as above.
**Answer:**
```sql
WITH spend AS (
  SELECT p.category, c.customer_name, SUM(o.amount) AS total
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY p.category, c.customer_name),
ranked AS (
  SELECT category, customer_name, total,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY total DESC) AS rn,
    LEAD(total)  OVER (PARTITION BY category ORDER BY total DESC) AS runner_up
  FROM spend)
SELECT category, customer_name, total, total - COALESCE(runner_up,0) AS lead_over_runner_up
FROM ranked WHERE rn = 1;
```
**Expected output:**
| category | customer_name | total | lead_over_runner_up |
|--|--|--|--|
| electronics | Beta | 450 | 50 |
| books | Acme | 75 | 75 |
| home | Acme | 40 | 40 |
*(electronics: Beta 200+250=450 beats Acme 400 by 50.)*

**Practice P98.** For each department, the top earner and the gap to #2.
**Table — employees** (emp_name, dept_id, salary) · **departments** (dept_id, dept_name)
**Answer:**
```sql
WITH r AS (SELECT d.dept_name, e.emp_name, e.salary,
             ROW_NUMBER() OVER (PARTITION BY d.dept_id ORDER BY e.salary DESC) rn,
             LEAD(e.salary) OVER (PARTITION BY d.dept_id ORDER BY e.salary DESC) nxt
           FROM employees e JOIN departments d ON e.dept_id=d.dept_id)
SELECT dept_name, emp_name, salary, salary - COALESCE(nxt,0) AS lead_gap FROM r WHERE rn=1;
```

---

### Q99. Every (country, category, month) where revenue fell vs. the previous month, with % drop.
**Table — orders** (customer_id, product_id, order_date, amount) — full 6 rows.
**Tables — customers** (id→country), **products** (id→category).
**Answer:**
```sql
WITH monthly AS (
  SELECT c.country, p.category, DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY c.country, p.category, DATE_TRUNC('month', o.order_date)),
growth AS (
  SELECT country, category, mth, rev,
    LAG(rev) OVER (PARTITION BY country, category ORDER BY mth) AS prev_rev FROM monthly)
SELECT country, category, mth, rev, prev_rev,
  (rev - prev_rev) * 100.0 / NULLIF(prev_rev,0) AS mom_pct
FROM growth WHERE prev_rev IS NOT NULL AND rev < prev_rev;
```
**Expected output:**
| country | category | mth | rev | prev_rev | mom_pct |
|--|--|--|--|--|--|
| US | electronics | 2026-03 | 250 | 600 | -58.3 |
*(US electronics: Jan 600 → Mar 250. LAG steps to the previous existing row, so Feb's absence means Mar's "prev" is Jan.)*

**Practice P99.** Creators whose monthly views dropped vs. the prior month.
**Table — video_events** (creator_id, event_type, event_time) · **creators** (id→name)
**Answer:**
```sql
WITH m AS (SELECT cr.creator_name, DATE_TRUNC('month',v.event_time) mth, COUNT(*) views
           FROM video_events v JOIN creators cr ON v.creator_id=cr.creator_id
           WHERE v.event_type='view' GROUP BY cr.creator_name, DATE_TRUNC('month',v.event_time)),
g AS (SELECT creator_name, mth, views, LAG(views) OVER (PARTITION BY creator_name ORDER BY mth) prev FROM m)
SELECT * FROM g WHERE prev IS NOT NULL AND views < prev;
```

---

### Q100. Products whose revenue rank within their category DROPPED month-over-month. (the no-nesting trap)
**Table — orders** (product_id, order_date, amount) + **products** (id→category).
**Answer:**
```sql
WITH m AS (
  SELECT p.category, p.product_id, p.product_name, DATE_TRUNC('month', o.order_date) AS mth, SUM(o.amount) AS rev
  FROM orders o JOIN products p ON o.product_id = p.product_id
  GROUP BY p.category, p.product_id, p.product_name, DATE_TRUNC('month', o.order_date)),
ranked AS (                       -- STEP 1: compute the rank
  SELECT category, product_id, product_name, mth, rev,
    RANK() OVER (PARTITION BY category, mth ORDER BY rev DESC) AS rnk FROM m),
with_prev AS (                    -- STEP 2: LAG the rank in a SEPARATE CTE
  SELECT category, product_name, mth, rnk,
    LAG(rnk) OVER (PARTITION BY category, product_id ORDER BY mth) AS prev_rnk FROM ranked)
SELECT category, product_name, mth, prev_rnk AS rank_last_month, rnk AS rank_this_month
FROM with_prev WHERE prev_rnk IS NOT NULL AND rnk > prev_rnk;
```
**Expected output:** *(empty on this data — no product's rank worsens)*
*The whole point: you CANNOT write `LAG(RANK() OVER(...)) OVER(...)` — nesting windows is a syntax error. Compute the rank in `ranked`, then LAG it in `with_prev`. To see a row, add an electronics product that outsells Gizmo in March.*

**Practice P100.** Employees whose salary rank within their department slipped after the latest raise cycle.
**Table — salary_history** (emp_id, dept_id, cycle, salary)
**Answer:**
```sql
WITH r AS (SELECT emp_id, dept_id, cycle, salary,
             RANK() OVER (PARTITION BY dept_id, cycle ORDER BY salary DESC) rnk FROM salary_history),
p AS (SELECT emp_id, dept_id, cycle, rnk, LAG(rnk) OVER (PARTITION BY emp_id ORDER BY cycle) prev FROM r)
SELECT emp_id, dept_id, cycle, prev AS rank_before, rnk AS rank_after
FROM p WHERE prev IS NOT NULL AND rnk > prev;
```

---

### Q101. Each customer's 3rd order — product and days since their 2nd order.
**Table — orders**
| order_id | customer_id | product_id | order_date |
|--|--|--|--|
| 1 | 101 | 1 | 2026-01-10 |
| 2 | 101 | 3 | 2026-02-05 |
| 5 | 101 | 4 | 2026-03-01 |
| 3 | 102 | 1 | 2026-01-20 |
| 6 | 102 | 2 | 2026-03-12 |

**Tables — customers** (id→name), **products** (id→name).
**Answer:**
```sql
WITH ord AS (
  SELECT o.customer_id, c.customer_name, o.order_date, p.product_name,
    ROW_NUMBER() OVER (PARTITION BY o.customer_id ORDER BY o.order_date)      AS seq,
    LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_date
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id)
SELECT customer_name, order_date AS third_order_date, product_name,
  DATEDIFF(order_date, prev_date) AS days_since_second
FROM ord WHERE seq = 3;
```
**Expected output:**
| customer_name | third_order_date | product_name | days_since_second |
|--|--|--|--|
| Acme | 2026-03-01 | Desk Lamp | 24 |
*(Only Acme has a 3rd order; Beta has 2, others 1.)*

**Practice P101.** Each user's 2nd session and the hours since their 1st.
**Table — sessions** (user_id, login_time)
**Answer:**
```sql
WITH s AS (SELECT user_id, login_time,
             ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY login_time) seq,
             LAG(login_time) OVER (PARTITION BY user_id ORDER BY login_time) prev FROM sessions)
SELECT user_id, login_time AS second_session,
  (UNIX_TIMESTAMP(login_time)-UNIX_TIMESTAMP(prev))/3600.0 AS hours_since_first
FROM s WHERE seq = 2;
```

---

### Q102. Orders where the customer switched to a different category than their previous order.
**Table — orders** (customer_id, product_id, order_date) + **products** (id→category) + **customers** (id→name).
**Answer:**
```sql
WITH ord AS (
  SELECT o.customer_id, c.customer_name, o.order_date, p.category,
    LAG(p.category)   OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_category,
    LAG(o.order_date) OVER (PARTITION BY o.customer_id ORDER BY o.order_date) AS prev_date
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id)
SELECT customer_name, order_date, category, prev_category,
  DATEDIFF(order_date, prev_date) AS days_since_prev
FROM ord WHERE prev_category IS NOT NULL AND category <> prev_category;
```
**Expected output:**
| customer_name | order_date | category | prev_category | days_since_prev |
|--|--|--|--|--|
| Acme | 2026-02-05 | books | electronics | 26 |
| Acme | 2026-03-01 | home | books | 24 |
*(Beta stayed in electronics → no switch. Lesson: LAG can pull a column from a *joined* table.)*

**Practice P102.** Sessions where the user switched device vs. their previous session.
**Table — sessions** (user_id, login_time, device)
**Answer:**
```sql
WITH s AS (SELECT user_id, login_time, device,
             LAG(device) OVER (PARTITION BY user_id ORDER BY login_time) prev_dev FROM sessions)
SELECT user_id, login_time, device, prev_dev FROM s WHERE prev_dev IS NOT NULL AND device <> prev_dev;
```

---

### Q103. Customers who spent more than the average customer in their country AND bought 3+ distinct categories.
**Table — orders** (customer_id, product_id, amount) + **customers** (id→name,country) + **products** (id→category).
**Answer:**
```sql
WITH spend AS (
  SELECT c.customer_id, c.customer_name, c.country,
    SUM(o.amount) AS total, COUNT(DISTINCT p.category) AS categories
  FROM orders o
  JOIN customers c ON o.customer_id = c.customer_id
  JOIN products  p ON o.product_id  = p.product_id
  GROUP BY c.customer_id, c.customer_name, c.country),
flagged AS (SELECT *, AVG(total) OVER (PARTITION BY country) AS country_avg FROM spend)
SELECT customer_name, country, total, categories
FROM flagged WHERE total > country_avg AND categories >= 3;
```
**Expected output:**
| customer_name | country | total | categories |
|--|--|--|--|
| Acme | US | 515 | 3 |
*(US avg = (Acme 515 + Beta 450)/2 = 482.5; Acme clears it and has 3 categories.)*

**Practice P103.** Creators with above-region-average views who posted in 3+ distinct months.
**Table — video_events** (creator_id, event_time) + **creators** (id→name,region)
**Answer:**
```sql
WITH s AS (SELECT cr.creator_id, cr.creator_name, cr.region, COUNT(*) views,
             COUNT(DISTINCT DATE_TRUNC('month', v.event_time)) active_months
           FROM video_events v JOIN creators cr ON v.creator_id=cr.creator_id
           GROUP BY cr.creator_id, cr.creator_name, cr.region),
f AS (SELECT *, AVG(views) OVER (PARTITION BY region) reg_avg FROM s)
SELECT creator_name, region, views, active_months FROM f WHERE views > reg_avg AND active_months >= 3;
```

---

### Q104. THE MONSTER — top-5 creators per region-month with share, MoM growth, rank, and rank movement.
**Table — video_events** (creator_id, event_type, event_time) + **creators** (id→name, region).
**Data (view events):** NovaK(US): Jan ×1, Feb ×2; LeoM(US): Feb ×1; RiyaG(IN): Jan ×1.
**Answer:**
```sql
WITH cm AS (
  SELECT cr.region, cr.creator_id, cr.creator_name,
    DATE_TRUNC('month', v.event_time) AS mth, COUNT(*) AS views
  FROM video_events v JOIN creators cr ON v.creator_id = cr.creator_id
  WHERE v.event_type = 'view'
  GROUP BY cr.region, cr.creator_id, cr.creator_name, DATE_TRUNC('month', v.event_time)),
enriched AS (
  SELECT region, creator_id, creator_name, mth, views,
    views * 100.0 / SUM(views) OVER (PARTITION BY region, mth)        AS pct_of_region,
    RANK()     OVER (PARTITION BY region, mth ORDER BY views DESC)     AS rnk,
    LAG(views) OVER (PARTITION BY region, creator_id ORDER BY mth)     AS prev_views
  FROM cm),
movement AS (
  SELECT region, creator_name, mth, views, pct_of_region, rnk, prev_views,
    LAG(rnk) OVER (PARTITION BY region, creator_id ORDER BY mth)       AS prev_rnk
  FROM enriched)
SELECT region, mth, rnk, creator_name, views,
  ROUND(pct_of_region,1) AS pct_of_region,
  ROUND((views - prev_views)*100.0/NULLIF(prev_views,0),1) AS mom_growth_pct,
  CASE WHEN prev_rnk IS NULL THEN 'new' WHEN rnk<prev_rnk THEN 'up'
       WHEN rnk>prev_rnk THEN 'down' ELSE 'same' END AS rank_movement
FROM movement WHERE rnk <= 5 ORDER BY region, mth, rnk;
```
**Expected output:**
| region | mth | rnk | creator_name | views | pct_of_region | mom_growth_pct | rank_movement |
|--|--|--|--|--|--|--|--|
| IN | 2026-01 | 1 | RiyaG | 1 | 100.0 | (null) | new |
| US | 2026-01 | 1 | NovaK | 1 | 100.0 | (null) | new |
| US | 2026-02 | 1 | NovaK | 2 | 66.7 | 100.0 | same |
| US | 2026-02 | 2 | LeoM | 1 | 33.3 | (null) | new |
*Lesson — four windows cooperate in `enriched` (share via SUM OVER, rank via RANK, growth via LAG(views)); but `LAG(rnk)` needs the rank to already exist, so it lives in the next CTE `movement`. That split IS the architecture.*

**Practice P104.** Same shape for retail: top-5 products per category-month with share, MoM growth, rank, movement.
**Table — orders** (product_id, order_date, amount) + **products** (id→name, category)
**Answer:**
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

## How to devise any boss query — four sentences before any SQL
1. **Base grain?** → the join + GROUP BY (one row per region-creator-month).
2. **Rank/compare within a group?** → window funcs (RANK, ROW_NUMBER, SUM OVER, AVG OVER).
3. **Need another row?** → LAG/LEAD — and if it's "LAG a rank," that's a *separate* CTE (windows can't nest).
4. **Final filter?** → rn=1, rnk<=N, "changed", "declined" — in the outer SELECT.

Write those four as English, and the CTE chain writes itself. Practice loop for every question above: read it → find the rows → hand-compute the answer → write SQL → check against the expected output.
