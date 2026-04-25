# Behavioral Analytics for Student Self-Discipline (CAPIR Framework)

This repository contains a Python implementation of the behavioral analytics pipeline proposed in the academic paper: **"Behavioral Analytics for Quantifying Student Self-Discipline in Learning Management Systems"** by Vanshika and Surya Madhavi (Chitkara University).

## Overview

The project processes Learning Management System (LMS) interaction logs to objectively quantify student self-regulation and predict academic risk using the **CAPIR** framework:
* **Connectivity (C):** Overall integration with the learning environment.
* **Acquisition (A):** Conversion of content exposure into demonstrable knowledge.
* **Productivity (P):** Quality of submitted independent and collaborative work.
* **Interactivity (I):** Engagement with the learning community.
* **Reactivity (R):** Latency between resource publication and first access.

The pipeline performs $K$-Means clustering to identify behavioral personas (Active Learners, Passive Learners, Socially Engaged, Academically Focused) and trains a Support Vector Machine (SVM) utilizing SMOTE to power an Early Warning System (EWS) for at-risk students.

## Project Structure

* `main.py`: The core executable script containing the data generation, feature engineering, clustering, and predictive modeling logic.
* `README.md`: Project documentation and setup instructions.

## Prerequisites

Ensure you have Python 3.8+ installed. You will need the following Python libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn