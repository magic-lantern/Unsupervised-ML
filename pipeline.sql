

@transform_pandas(
    Output(rid="ri.vector.main.execute.0309ab31-0e32-4ed5-9fac-f203b2048224"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
SELECT *
FROM inpatient_ml_dataset

