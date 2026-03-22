## Data Source

The raw Instacart dataset is not included in this repository due to file size limitations.

To reproduce the pipeline:
1. Download the Instacart Online Grocery Shopping Dataset from Kaggle
2. Place the following CSV files under `data/raw/`:
   - orders.csv
   - order_products__prior.csv
   - order_products__train.csv
   - products.csv
   - aisles.csv
   - departments.csv

If you only want to run a smoke test without downloading data:

```bash
python run_pipeline.py --use-sample-data-if-missing
```
