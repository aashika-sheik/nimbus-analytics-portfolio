"""
Synthetic D2C Shopify Dataset Generator
----------------------------------------
Generates a realistic 18-month dataset for a fictional D2C skincare brand
("Nimbus Skincare") to be used across three portfolio projects:
  1. Power BI  -> CAC / ROAS / LTV dashboard
  2. Tableau   -> Cohort retention + RFM segmentation story
  3. Python ML -> Churn prediction + SHAP-based action recommender

Design choices (intentional, not arbitrary):
- 18 months of history so cohort curves and seasonality are visible
- 4 acquisition channels with different CAC/quality profiles (realistic D2C mix)
- Seasonality: spikes around Nov-Dec (BFCM/holiday) and a smaller spike in summer
- Discount usage varies by channel (paid social users are more discount-sensitive)
- Built-in churn signal: customers who haven't ordered in 90+ days and show
  declining engagement events are the "true" churners for the ML label
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random

# ----------------------------
# CONFIG
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
fake = Faker()
Faker.seed(SEED)

START_DATE = datetime(2024, 2, 1)
END_DATE = datetime(2025, 7, 31)  # 18 months
TOTAL_DAYS = (END_DATE - START_DATE).days

N_CUSTOMERS = 6000

CHANNELS = {
    # channel: (weight/share of new customers, base_cac, discount_sensitivity, repeat_purchase_prob)
    "Meta Ads":        {"share": 0.35, "cac": 18, "discount_sens": 0.55, "repeat_prob": 0.28},
    "Google Ads":      {"share": 0.20, "cac": 24, "discount_sens": 0.35, "repeat_prob": 0.35},
    "Influencer":      {"share": 0.15, "cac": 12, "discount_sens": 0.45, "repeat_prob": 0.40},
    "Organic/SEO":     {"share": 0.15, "cac": 3,  "discount_sens": 0.20, "repeat_prob": 0.50},
    "Email/Referral":  {"share": 0.15, "cac": 5,  "discount_sens": 0.25, "repeat_prob": 0.55},
}

COUNTRIES = ["UAE", "Saudi Arabia", "India", "UK", "USA", "Kuwait", "Qatar"]
COUNTRY_WEIGHTS = [0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05]

AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_WEIGHTS = [0.20, 0.38, 0.25, 0.12, 0.05]

PRODUCTS = [
    ("Vitamin C Serum", 22),
    ("Hydrating Moisturizer", 28),
    ("Niacinamide Serum", 19),
    ("SPF 50 Sunscreen", 24),
    ("Retinol Night Cream", 32),
    ("Gentle Cleanser", 16),
    ("Eye Cream", 26),
    ("Face Mask Set", 20),
]

def seasonal_multiplier(date):
    """Boost demand around BFCM (Nov) and a smaller summer spike (Jun)."""
    m = date.month
    if m == 11:
        return 2.2
    if m == 12:
        return 1.6
    if m == 6:
        return 1.3
    if m in (1, 8):
        return 0.85
    return 1.0

def weighted_choice(options, weights):
    return random.choices(options, weights=weights, k=1)[0]

# ----------------------------
# 1. CUSTOMERS
# ----------------------------
print("Generating customers...")
customers = []
# distribute signups across the period with seasonality weighting
day_weights = [seasonal_multiplier(START_DATE + timedelta(days=d)) for d in range(TOTAL_DAYS)]
day_weights = np.array(day_weights) / sum(day_weights)
signup_day_offsets = np.random.choice(range(TOTAL_DAYS), size=N_CUSTOMERS, p=day_weights)

channel_names = list(CHANNELS.keys())
channel_shares = [CHANNELS[c]["share"] for c in channel_names]

for i in range(N_CUSTOMERS):
    customer_id = f"CUST{i+1:05d}"
    signup_date = START_DATE + timedelta(days=int(signup_day_offsets[i]))
    channel = weighted_choice(channel_names, channel_shares)
    country = weighted_choice(COUNTRIES, COUNTRY_WEIGHTS)
    age_band = weighted_choice(AGE_BANDS, AGE_WEIGHTS)
    gender = weighted_choice(["Female", "Male", "Other"], [0.78, 0.20, 0.02])

    customers.append({
        "customer_id": customer_id,
        "signup_date": signup_date.date(),
        "acquisition_channel": channel,
        "country": country,
        "age_band": age_band,
        "gender": gender,
    })

customers_df = pd.DataFrame(customers)

# ----------------------------
# 2. ORDERS
# ----------------------------
print("Generating orders...")
orders = []
order_counter = 1

for _, cust in customers_df.iterrows():
    channel_cfg = CHANNELS[cust["acquisition_channel"]]
    signup_dt = datetime.combine(cust["signup_date"], datetime.min.time())

    # First order: happens near signup (within 0-3 days), ~92% convert
    if random.random() < 0.92:
        first_order_date = signup_dt + timedelta(days=random.randint(0, 3))
        n_orders_for_customer = 1

        # simulate repeat purchases via geometric-ish process based on channel repeat_prob
        current_date = first_order_date
        while True:
            if current_date > END_DATE:
                break
            gap_days = int(np.random.exponential(scale=55))  # avg ~55 days between repeat orders
            gap_days = max(14, gap_days)
            next_date = current_date + timedelta(days=gap_days)
            if next_date > END_DATE:
                break
            if random.random() < channel_cfg["repeat_prob"]:
                current_date = next_date
                n_orders_for_customer += 1
            else:
                break
            if n_orders_for_customer >= 12:  # cap
                break

        # generate the actual order rows
        order_date = first_order_date
        for order_num in range(n_orders_for_customer):
            product_name, base_price = random.choice(PRODUCTS)
            qty = random.choices([1, 2, 3], weights=[0.65, 0.25, 0.10])[0]

            discount_used = 0
            if random.random() < channel_cfg["discount_sens"]:
                discount_used = random.choice([10, 15, 20, 25])  # % off

            gross = base_price * qty
            revenue = round(gross * (1 - discount_used / 100), 2)

            orders.append({
                "order_id": f"ORD{order_counter:06d}",
                "customer_id": cust["customer_id"],
                "order_date": order_date.date(),
                "product": product_name,
                "quantity": qty,
                "discount_pct": discount_used,
                "revenue": revenue,
                "is_first_order": order_num == 0,
            })
            order_counter += 1

            if order_num < n_orders_for_customer - 1:
                gap_days = max(14, int(np.random.exponential(scale=55)))
                order_date = order_date + timedelta(days=gap_days)
                if order_date > END_DATE:
                    break

orders_df = pd.DataFrame(orders)

# ----------------------------
# 3. CAMPAIGNS (daily spend/impressions/clicks by channel)
# ----------------------------
print("Generating campaign spend...")
campaigns = []
for d in range(TOTAL_DAYS):
    date = START_DATE + timedelta(days=d)
    season_mult = seasonal_multiplier(date)
    for channel in channel_names:
        cfg = CHANNELS[channel]
        if channel in ("Organic/SEO", "Email/Referral"):
            # low/no media spend channels, but still track "cost" (content/tools/CRM cost)
            spend = round(np.random.uniform(5, 20) * season_mult, 2)
        else:
            base_daily_spend = {
                "Meta Ads": 180, "Google Ads": 150, "Influencer": 90
            }[channel]
            spend = round(base_daily_spend * season_mult * np.random.uniform(0.7, 1.3), 2)

        impressions = int(spend * np.random.uniform(80, 140))
        ctr = np.random.uniform(0.01, 0.035)
        clicks = int(impressions * ctr)

        campaigns.append({
            "date": date.date(),
            "channel": channel,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
        })

campaigns_df = pd.DataFrame(campaigns)

# ----------------------------
# 4. EVENTS (engagement signals used for churn labeling/features)
# ----------------------------
print("Generating engagement events...")
events = []
event_types_weights = {
    "site_visit": 0.45,
    "cart_add": 0.20,
    "cart_abandon": 0.15,
    "email_open": 0.12,
    "review_submitted": 0.05,
    "support_ticket": 0.03,
}

for _, cust in customers_df.iterrows():
    signup_dt = datetime.combine(cust["signup_date"], datetime.min.time())
    cust_orders = orders_df[orders_df["customer_id"] == cust["customer_id"]]

    if len(cust_orders) > 0:
        last_order_date = pd.to_datetime(cust_orders["order_date"]).max()
    else:
        last_order_date = signup_dt

    # active window: events cluster around signup and around order dates, tapering off after last order
    days_since_last_order = (END_DATE - last_order_date).days
    # engagement decays the longer since last order (this is the churn signal)
    n_events = max(1, int(np.random.poisson(lam=8) * np.exp(-days_since_last_order / 180)))

    for _ in range(n_events):
        event_date = signup_dt + timedelta(days=random.randint(0, max(1, (END_DATE - signup_dt).days)))
        if event_date > END_DATE:
            continue
        etype = weighted_choice(list(event_types_weights.keys()), list(event_types_weights.values()))
        events.append({
            "customer_id": cust["customer_id"],
            "event_date": event_date.date(),
            "event_type": etype,
        })

events_df = pd.DataFrame(events)

# ----------------------------
# SAVE
# ----------------------------
import os
OUT_DIR = "/home/claude/nimbus_data"
os.makedirs(OUT_DIR, exist_ok=True)

customers_df.to_csv(f"{OUT_DIR}/customers.csv", index=False)
orders_df.to_csv(f"{OUT_DIR}/orders.csv", index=False)
campaigns_df.to_csv(f"{OUT_DIR}/campaigns.csv", index=False)
events_df.to_csv(f"{OUT_DIR}/events.csv", index=False)

print("\n--- SUMMARY ---")
print(f"Customers: {len(customers_df):,}")
print(f"Orders: {len(orders_df):,}  |  Total revenue: AED {orders_df['revenue'].sum():,.0f}")
print(f"Campaign rows: {len(campaigns_df):,}  |  Total spend: AED {campaigns_df['spend'].sum():,.0f}")
print(f"Event rows: {len(events_df):,}")
print(f"\nDate range: {START_DATE.date()} to {END_DATE.date()}")
print(f"Files saved to: {OUT_DIR}")
