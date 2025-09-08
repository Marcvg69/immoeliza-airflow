from immoeliza.modeling.train_regression import train
def run(**context):
    # pull path returned by clean_for_training via XCom if you want; here we allow DAG conf override
    training_path = (getattr(context.get("dag_run"), "conf", {}) or {}).get("training_path")
    if not training_path:
        # read latest by day from data/training
        import glob, os
        matches = sorted(glob.glob("data/training/*/training.parquet"))
        if not matches:
            raise FileNotFoundError("No training.parquet found. Run clean_for_training first.")
        training_path = matches[-1]
    return train(training_path)
