import pandas as pd
from .config import DATA_DIR

def extract_raw():
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    opp = pd.read_csv(DATA_DIR / "order_products__prior.csv")
    opt = pd.read_csv(DATA_DIR / "order_products__train.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    aisles = pd.read_csv(DATA_DIR / "aisles.csv")
    departments = pd.read_csv(DATA_DIR / "departments.csv")
    return orders, opp, opt, products, aisles, departments
