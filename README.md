# Data Engineering Project

This repository contains a multi-milestone data engineering project built around a fintech loans dataset.

## Repository Structure

- `Milestone 1/` – initial exploration, cleaning, and dataset preparation in notebooks.
- `Milestone 2/` – containerized ETL components with PostgreSQL and Kafka.
- `Milestone 3/` – Spark-based processing and feature engineering outputs.
- `Milestone 4/` – Apache Airflow ETL orchestration, PostgreSQL loading, and Streamlit dashboarding.

## Main Technologies

- Python (pandas, scikit-learn)
- Apache Spark
- Apache Airflow
- PostgreSQL
- Kafka
- Streamlit + Plotly
- Docker / Docker Compose

## Quick Start

### Milestone 2

```bash
cd "Milestone 2"
docker compose up --build
```

### Milestone 4

```bash
cd "Milestone 4"
docker compose up --build
```

Then open:

- Airflow UI: `http://localhost:8080`
- (Milestone 2) pgAdmin: `http://localhost:8090`

## Notes

- Dataset files and milestone descriptions are included in the corresponding milestone folders.
- This repository is organized as coursework deliverables, so each milestone is mostly self-contained.
