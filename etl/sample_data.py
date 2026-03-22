from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_instacart_data(
    num_users: int = 80,
    num_products: int = 60,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)

    product_ids = np.arange(1, num_products + 1)
    aisle_ids = np.arange(1, 11)
    department_ids = np.arange(1, 6)

    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": [f"product_{i}" for i in product_ids],
            "aisle_id": rng.choice(aisle_ids, size=num_products),
            "department_id": rng.choice(department_ids, size=num_products),
        }
    )
    aisles = pd.DataFrame({"aisle_id": aisle_ids, "aisle": [f"aisle_{i}" for i in aisle_ids]})
    departments = pd.DataFrame(
        {"department_id": department_ids, "department": [f"department_{i}" for i in department_ids]}
    )

    orders_rows = []
    prior_rows = []
    train_rows = []
    order_id = 1

    for user_id in range(1, num_users + 1):
        user_pool_size = int(rng.integers(8, min(20, num_products) + 1))
        user_product_pool = rng.choice(product_ids, size=user_pool_size, replace=False)
        prior_order_count = int(rng.integers(5, 12))

        purchase_counts = {int(pid): 0 for pid in user_product_pool}

        for order_number in range(1, prior_order_count + 1):
            orders_rows.append(
                {
                    "order_id": order_id,
                    "user_id": user_id,
                    "eval_set": "prior",
                    "order_number": order_number,
                    "order_dow": int(rng.integers(0, 7)),
                    "order_hour_of_day": int(rng.integers(6, 23)),
                    "days_since_prior_order": np.nan if order_number == 1 else int(rng.integers(1, 21)),
                }
            )

            item_count = int(rng.integers(3, min(12, user_pool_size) + 1))
            chosen_products = rng.choice(user_product_pool, size=item_count, replace=False)

            for add_to_cart_order, product_id in enumerate(chosen_products, start=1):
                pid = int(product_id)
                reordered = 1 if purchase_counts[pid] > 0 else 0
                prior_rows.append(
                    {
                        "order_id": order_id,
                        "product_id": pid,
                        "add_to_cart_order": add_to_cart_order,
                        "reordered": reordered,
                    }
                )
                purchase_counts[pid] += 1

            order_id += 1

        train_order_number = prior_order_count + 1
        orders_rows.append(
            {
                "order_id": order_id,
                "user_id": user_id,
                "eval_set": "train",
                "order_number": train_order_number,
                "order_dow": int(rng.integers(0, 7)),
                "order_hour_of_day": int(rng.integers(6, 23)),
                "days_since_prior_order": int(rng.integers(1, 21)),
            }
        )

        train_item_count = int(rng.integers(3, min(10, user_pool_size) + 1))
        cold_pool = np.setdiff1d(product_ids, user_product_pool)
        cold_count = int(min(max(4, user_pool_size // 2), len(cold_pool)))
        if cold_count > 0:
            cold_products = rng.choice(cold_pool, size=cold_count, replace=False)
            train_candidate_pool = np.concatenate([user_product_pool, cold_products])
        else:
            train_candidate_pool = user_product_pool

        # Keep warm products predictive but include enough cold products so smoke-test metrics are realistic.
        base_weights = []
        for pid in train_candidate_pool:
            pid_int = int(pid)
            if pid_int in purchase_counts:
                base_weights.append(1.0 + 0.15 * purchase_counts[pid_int])
            else:
                base_weights.append(1.2)
        base_weights = np.asarray(base_weights, dtype=float)
        base_weights /= base_weights.sum()
        train_item_count = min(train_item_count, len(train_candidate_pool))
        train_products = rng.choice(
            train_candidate_pool,
            size=train_item_count,
            replace=False,
            p=base_weights,
        )

        for add_to_cart_order, product_id in enumerate(train_products, start=1):
            pid = int(product_id)
            reordered = 1 if purchase_counts.get(pid, 0) > 0 else 0
            if rng.random() < 0.05:
                reordered = 1 - reordered
            train_rows.append(
                {
                    "order_id": order_id,
                    "product_id": pid,
                    "add_to_cart_order": add_to_cart_order,
                    "reordered": reordered,
                }
            )

        order_id += 1

    orders = pd.DataFrame(orders_rows)
    order_products_prior = pd.DataFrame(prior_rows)
    order_products_train = pd.DataFrame(train_rows)

    return (
        orders,
        order_products_prior,
        order_products_train,
        products,
        aisles,
        departments,
    )
