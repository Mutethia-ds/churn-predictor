# Great Expectations

The expectation suite is defined in code at
`src/churn/data/validate_data.py` (using GE's ephemeral/fluent API) rather
than as static JSON here, so it's easy to read and version alongside the
pipeline code it protects.

Run it directly:

```bash
python -m churn.data.validate_data
```

It exits with a non-zero status code if any expectation fails, which is
what makes it usable as a CI/Airflow gate.

If you'd prefer a full Great Expectations project (with a persistent
Data Context, HTML "Data Docs", and a checkpoint you can point Airflow's
`GreatExpectationsOperator` at), run `great_expectations init` in this
folder and migrate the expectations from `validate_data.py` into it.
