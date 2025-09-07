# 🧠  Multi-Agent System with SmolAgents
# 📌  Overview

## This project implements a multi-agent system built on top of SmolAgents

It integrates specialized agents for research, code execution, and data science workflows under the coordination of a manager agent.
The system demonstrates how autonomous agents can collaborate to solve end-to-end tasks such as:
**Web research**
**Writing and executing code safely in sandboxed Docker environments**
**Data analysis, cleaning, visualization, and machine learning via AutoML**


## 🤖 Agents

### 🗂️ Planner Agent

**Breaks down complex tasks into subtasks**
**Routes them to the appropriate specialized agents**
**Uses Hugging Face Qwen2.5-Coder-32B-Instruct model for reasoning**


### 🔎 Research Agent

**Performs internet searches using the Tavily API**
**Extracts contextual insights with query/topic filtering**
**Supports raw content retrieval for deeper analysis**


### 💻 Coding Agent

Executes Python code in sandboxed Docker containers.
Tools supported:

**code_executor → Safe Docker execution**
**run_manifest → End-to-end code execution with dependency installation**
**save_files, list_workspace_files, package_artifact, cleanup_workspace**
**Supports artifact packaging (ZIP) for deployment**


### 📊 Data Science Agent

Provides a full data science workflow:
**LoadData: Reads CSVs into memory**
**CleanData: Handles missing values, encodes categoricals, and removes outliers**
**EDA: Produces summary statistics, correlations, skewness, distributions & recommendations**
**AutoML (classification & regression)**

## 🔧 Features

**🔄 Orchestration with a Manager Agent that routes tasks to the correct specialized agent**
**🌐 Internet Research with TavilyClient**
**🐳 Secure Code Execution with Docker sandboxes (no host pollution)**
**📈 Automated EDA with advanced visualizations (Seaborn/Matplotlib)**
**🤖 AutoML baseline training with scikit-learn (classification & regression)**
**📦 Artifact Management: Save, list, package, and clean workspaces**
**🔌 Extensible: Add more tools/agents easily**



## 🛠️ Example Scenarios

### Research + Analysis

“Find recent developments in AI hardware and summarize their impact on model training efficiency.”
→ Research Agent fetches news → Data Science Agent correlates with dataset.

### Code Execution

“Write a Python script that generates synthetic data and plot distributions.”
→ Planner delegates → Coder executes safely in Docker → Artifact is packaged.

### AutoML Pipeline

“Upload titanic.csv and predict survival.”
→ Data Science Agent loads, cleans, performs EDA, trains models, and returns metrics.
AutoML: Detects target, selects between classification/regression, trains baseline models (Random Forest, Logistic/Linear Regression), evaluates metrics, and exports trained models
Generates plots (missingness heatmap, correlation heatmap, distributions, target relationships).


Streamlit UI
The UI is still work-in-progress
