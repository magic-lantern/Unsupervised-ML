

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.e3631af7-a612-48c9-8d3d-82cf3df238de"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
select count(1), year(visit_start_date) as s_year, month(visit_start_date) as s_month
from inpatient_ml_dataset
group by s_year, s_month
order by s_year, s_month

@transform_pandas(
    Output(rid="ri.foundry.main.dataset.87ae0fb9-7045-4180-a60a-2869682b6f8a"),
    inpatient_ml_dataset=Input(rid="ri.foundry.main.dataset.07927bca-b175-4775-9c55-a371af481cc1")
)
select count(1), year(visit_end_date) as s_year, month(visit_end_date) as s_month
from inpatient_ml_dataset
group by s_year, s_month
order by s_year, s_month

