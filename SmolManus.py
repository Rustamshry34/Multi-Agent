from tavily import TavilyClient 
import subprocess, tempfile, time, os
from pathlib import Path
from typing import Dict, Any, List, Literal
import shutil, zipfile
from uuid import uuid4
from smolagents import tool, CodeAgent, InferenceClientModel, ToolCallingAgent
# For Data Analysis Agent
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import io



# Initialize Tavily client for web search
#tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
tavily_client = TavilyClient(api_key="")
#os.environ["HF_TOKEN"] = "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"
os.environ["HF_TOKEN"] = ""
# ----------------- Tools -----------------
    
@tool
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance","science","technology","economy"] = "general",
    include_raw_content: bool = False,
)-> List[Dict[str, Any]]:
    
    """
    Tool to perform an internet search using the Tavily API.

    This tool allows the agent to gather information from the web
    based on a query and a specified topic. It returns a list of
    search results, optionally including the raw content of the
    webpages.

    Args:
        query (str): The search query or keywords to look up on the web.
        max_results (int, optional): Maximum number of search results to return. 
                                     Defaults to 5.
        topic (Literal["general", "news", "finance", "science", "technology", "economy"], optional): 
            Category of the search to prioritize relevant content. Defaults to "general".
        include_raw_content (bool, optional): If True, include the full raw content of the results; 
                                             otherwise, only metadata is returned. Defaults to False.

    Returns:
        List[Dict[str, Any]]: A list of search results from Tavily, with each item containing
                              relevant information such as title, URL, snippet, and optionally raw content.
    """

    result1 = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return result1

@tool
def code_executor(
    image: str,
    cmds: List[str],
    mounts: Dict[str, str] = None,
    host_workspace: str = None,
    container_workdir: str = "/workspace",
    timeout: int = 60,
    allow_network: bool = False,
) -> Dict[str, Any]:
    
    """
    Executes a sequence of shell commands inside a Docker container.

    This tool allows safe and isolated execution of code or scripts
    using a specified Docker image. It supports mounting host directories,
    custom working directories, timeout handling, and optional network access.

    Args:
        image (str): The Docker image to use for execution (e.g., "python:3.11-slim").
        cmds (List[str]): A list of shell commands to run inside the container.
        mounts (Dict[str, str], optional): Dictionary mapping host paths to container paths
                                           for volume mounting. Defaults to None.
        host_workspace (str, optional): Path on the host machine to use as workspace.
                                        If None, a temporary directory is created. Defaults to None.
        container_workdir (str, optional): Working directory inside the container. Defaults to "/workspace".
        timeout (int, optional): Maximum execution time in seconds before terminating the process. Defaults to 60.
        allow_network (bool, optional): Whether to allow network access inside the container.
                                        Defaults to False (safe default).

    Returns:
        Dict[str, Any]: A dictionary containing execution results:
            - stdout (str): Standard output from the container.
            - stderr (str): Standard error output.
            - exit_code (int): Exit code of the executed commands.
            - runtime_s (float): Execution time in seconds.
            - files (List[str]): List of files created in the host workspace (relative paths).
            - host_workspace (str): Path to the host workspace used for execution.

    Notes:
        - Ensures that the host workspace is always mounted to the container.
        - Normalizes Windows paths for Docker volume mounting.
        - Safely handles subprocess timeouts and captures output.
    """

    if host_workspace is None:
        host_workspace = tempfile.mkdtemp(prefix="mini_manus_ws_")
    # Ensure mounts include host_workspace -> container_workdir
    mounts = dict(mounts or {})
    if host_workspace not in mounts:
        mounts[host_workspace] = container_workdir

    docker_cmd = ["docker", "run", "--rm", "--memory", "512m", "--cpus", "1"]
    if not allow_network:
        docker_cmd += ["--network", "none"]

    # Normalize Windows backslashes -> forward slashes for docker -v on some setups
    def _norm(p: str) -> str:
        return p.replace("\\", "/")

    for host, cont in mounts.items():
        docker_cmd += ["-v", f"{_norm(host)}:{cont}"]

    docker_cmd += ["-w", container_workdir, image]
    joined = " && ".join(cmds) if cmds else "echo 'No commands provided'"
    docker_cmd += ["sh", "-lc", joined]

    start = time.time()
    try:
        proc = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
        runtime = time.time() - start

        # Gather files from the host workspace (NOT container path)
        files = []
        try:
            for p in Path(host_workspace).rglob("*"):
                if p.is_file():
                    files.append(str(p.relative_to(host_workspace)))
        except Exception:
            files = []

        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
            "runtime_s": round(runtime, 3),
            "files": files,
            "host_workspace": host_workspace,
        }
    except subprocess.TimeoutExpired as te:
        return {
            "stdout": te.stdout or "",
            "stderr": (te.stderr or "") + f"\n[Timed out after {timeout}s]",
            "exit_code": -1,
            "runtime_s": round(time.time() - start, 3),
            "files": [],
            "host_workspace": host_workspace,
        }
@tool
def save_files(manifest_files: List[Dict[str,str]], workspace: str = None) -> str:
    
    """
    Saves a list of files to a host workspace directory.

    This tool creates the specified files with their content on the host system.
    Each file is defined by a dictionary containing a relative path and content.
    If no workspace path is provided, a temporary directory is created automatically.

    Args:
        manifest_files (List[Dict[str, str]]): A list of file descriptors, 
            where each descriptor is a dictionary with:
            - "path" (str): Relative file path (e.g., "app.py" or "src/module.py").
            - "content" (str): The content to write into the file.
        workspace (str, optional): Path to the host directory where files should be saved.
                                   If None, a temporary directory is created. Defaults to None.

    Returns:
        str: The path to the host workspace directory where the files were saved.

    Notes:
        - Automatically creates parent directories if they do not exist.
        - Overwrites files if they already exist at the same path.
        - Useful for preparing workspaces for code execution in sandboxed environments.
    """

    if workspace is None:
        workspace = tempfile.mkdtemp(prefix="mini_manus_ws_")
    ws = Path(workspace)
    ws.mkdir(parents=True, exist_ok=True)
    for f in manifest_files:
        p = ws / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"], encoding="utf-8")
    return str(ws)

# 2) List files in a workspace (relative)
@tool
def list_workspace_files(workspace: str) -> List[str]:

    """
    Recursively list all files in a given workspace directory.

    This tool traverses the workspace directory and collects all file paths,
    returning them relative to the workspace root. It is useful for inspecting 
    the contents of a workspace, packaging artifacts, or tracking generated files.

    Args:
        workspace (str): Path to the workspace directory to list.

    Returns:
        List[str]: A list of file paths relative to the workspace root.

    Notes:
        - Only files are included; directories themselves are ignored.
        - If the workspace path is invalid or an error occurs during traversal,
          an empty list is returned.
        - Paths are returned as strings using forward slashes.
    """

    files = []
    try:
        for p in Path(workspace).rglob("*"):
            if p.is_file():
                files.append(str(p.relative_to(workspace)))
    except Exception:
        pass
    return files

# 3) Package artifact (zip) and return path
@tool
def package_artifact(workspace: str, out_dir: str = None) -> str:

    """
    Package the contents of a workspace directory into a ZIP archive.

    This tool collects all files within a given workspace and compresses 
    them into a single ZIP file, which can be used as an artifact for 
    deployment, sharing, or backup purposes.

    Args:
        workspace (str): Path to the workspace directory to package.
        out_dir (str, optional): Directory to save the generated ZIP file. 
            If None, a temporary directory will be created.

    Returns:
        str: Absolute file path of the created ZIP archive.

    Notes:
        - Only files are included in the ZIP archive; directories themselves 
          are not stored.
        - The ZIP filename is automatically generated using a UUID to ensure 
          uniqueness.
        - If `out_dir` does not exist, it will be created.
        - Useful for packaging code, data, or other artifacts generated 
          during automated workflows.
    """

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="mini_manus_artifacts_")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    zip_name = Path(out_dir) / f"artifact_{uuid4().hex}.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(workspace).rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(workspace))
    return str(zip_name)

# 4) Cleanup workspace
@tool
def cleanup_workspace(workspace: str, keep: bool = False) -> None:

    """
    Safely removes a workspace directory and all its contents.

    This tool is used to clean up temporary directories created during 
    code execution, testing, or file manipulation. It ensures that the 
    workspace is deleted unless explicitly preserved.

    Args:
        workspace (str): Path to the workspace directory to delete.
        keep (bool, optional): If True, the workspace will not be deleted.
            Defaults to False.

    Returns:
        None

    Notes:
        - Any errors during deletion (e.g., non-existent directory, permission issues) 
          are silently ignored.
        - Use `keep=True` to preserve the workspace, for example, when artifacts 
          need to be inspected after execution.
        - Intended for host-side cleanup of temporary directories used in containerized 
          or local code execution workflows.
    """

    if keep:
        return
    try:
        shutil.rmtree(workspace)
    except Exception:
        pass

# 5) Run a manifest end-to-end using your code_executor (uses Docker image + run_commands)
@tool
def run_manifest(manifest: Dict[str, Any], base_image: str = "python:3.11-slim", timeout: int = 120, keep_workspace: bool = False) -> Dict[str, Any]:
    
    """
    Executes a manifest of files and commands inside a Docker container and optionally packages the workspace.

    This tool automates the process of:
    1. Saving provided files to a host workspace.
    2. Installing dependencies (if a `requirements.txt` is present or if `install_libs` is specified).
    3. Running commands and optional test commands inside a Docker container.
       - Commands referencing workspace files are automatically adjusted to point to the container workspace.
    4. Collecting outputs, listing files, and optionally packaging the workspace into a ZIP artifact.
    5. Cleaning up the workspace unless `keep_workspace=True`.

    Args:
        manifest (Dict[str, Any]): A dictionary describing the manifest, with the following keys:
            - "files" (List[Dict[str,str]]): List of files to save, each with "path" and "content".
            - "run_commands" (List[str], optional): Commands to execute inside the container.
            - "test_command" (str, optional): A command for testing/verifying the execution.
            - "install_libs" (List[str], optional): A list of Python packages to install dynamically
              (e.g., ["crewai", "transformers"]). Installed before any run/test commands.
        base_image (str, optional): Docker image to use for execution. Defaults to "python:3.11-slim".
        timeout (int, optional): Maximum time in seconds for container execution. Defaults to 120.
        keep_workspace (bool, optional): If True, preserves the host workspace after execution. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing execution results and metadata:
            - "stdout" (str): Standard output from the execution.
            - "stderr" (str): Standard error from the execution.
            - "exit_code" (int): Exit code of the executed commands.
            - "runtime_s" (float): Total runtime in seconds.
            - "files" (List[str]): List of files present in the workspace after execution.
            - "artifact" (str or None): Path to a ZIP file of the workspace, if packaging succeeded.
            - "workspace" (str): Path to the host workspace.

    Notes:
        - If `requirements.txt` exists, dependencies are installed automatically inside the container.
        - If `install_libs` is provided, those packages are installed dynamically via pip.
        - Commands that reference workspace files are automatically adjusted to point to the container workspace.
        - Network access is enabled briefly during dependency installation.
        - Commands are executed sequentially inside the container.
        - Workspace cleanup is automatic unless `keep_workspace=True`.
        - Useful for safely running and testing code in isolated, reproducible environments.
    """

    files = manifest.get("files", [])
    run_cmds = manifest.get("run_commands", [])
    test_cmd = manifest.get("test_command")
    install_libs = manifest.get("install_libs", [])   # 👈 NEW
    host_workspace = save_files(files)  # this returns a host path

    # Map host workspace -> container path
    mounts = {host_workspace: "/workspace"}

    # Pre-install step if requirements.txt exists
    install_cmds = []
    if install_libs:
        # install arbitrary packages inside container
        libs = " ".join(install_libs)
        install_cmds.append(f"pip install {libs}")

    if (Path(host_workspace) / "requirements.txt").exists():
        install_cmds.append("pip install -r requirements.txt")

    #NEW 
    def fix_file_paths(cmds: List[str]) -> List[str]:
        fixed = []
        for c in cmds:
            parts = c.split()
            if parts[0] == "python" and len(parts) > 1:
                parts[1] = f"/workspace/{parts[1]}"
            fixed.append(" ".join(parts))
        return fixed
    

    # Build the full command sequence (run installs first if present)

    run_cmds = fix_file_paths(run_cmds)
    if test_cmd:
        test_cmd = fix_file_paths([test_cmd])[0]

    # Build full command list
    cmds = install_cmds + [f"cd /workspace && {c}" for c in run_cmds]
    if test_cmd:
        cmds.append(f"cd /workspace && {test_cmd}")

    if not cmds:
        cmds = ["cd /workspace && echo 'No commands provided'"]


    # If we're installing requirements, allow network briefly (set allow_network=True)
    allow_network = bool(install_cmds)

    exec_res = code_executor(
        image=base_image,
        cmds=cmds,
        mounts=mounts,
        host_workspace=host_workspace,
        container_workdir="/workspace",
        timeout=timeout,
        allow_network=allow_network,
    )

    # gather host-side file list (relative)
    files_list = list_workspace_files(host_workspace)

    # package artifact (optional)
    artifact = None
    try:
        artifact = package_artifact(host_workspace)
    except Exception:
        artifact = None

    result = {
        "stdout": exec_res.get("stdout", ""),
        "stderr": exec_res.get("stderr", ""),
        "exit_code": exec_res.get("exit_code", 1),
        "runtime_s": exec_res.get("runtime_s", None),
        "files": files_list,
        "artifact": artifact,
        "workspace": host_workspace,
    }

    # decide whether to cleanup workspace
    cleanup_workspace(host_workspace, keep=keep_workspace)
    return result

def detect_target_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the most likely target column based on naming, cardinality, and type.
    """
    if df.empty or len(df.columns) < 2:
        return None

    scores = {}

    for col in df.columns:
        score = 0.0
        name_lower = col.lower()

        # Rule 1: Name matches common target keywords
        keywords = ["target", "label", "class", "outcome", "result", "y", "output", "flag", "status", "churn", "survived", "price", "sale"]
        if any(kw in name_lower for kw in keywords):
            score += 3.0
        if name_lower in ["target", "label", "class", "y"]:
            score += 2.0

        # Rule 2: Binary or low-cardinality categorical → likely classification
        nunique = df[col].nunique()
        total = len(df)
        unique_ratio = nunique / total

        if nunique == 2 and df[col].dtype in ["int64", "object", "category"]:
            score += 4.0  # Strong signal
        elif nunique <= 20 and df[col].dtype in ["int64", "object", "category"]:
            score += 3.0

        # Rule 3: High unique ratio + numeric → likely regression target
        if unique_ratio > 0.8 and df[col].dtype in ["int64", "float64"]:
            score += 2.5

        # Rule 4: Avoid ID-like or high-cardinality text
        id_keywords = ["id", "name", "email", "phone", "address", "username", "url", "link"]
        if any(kw in name_lower for kw in id_keywords):
            score -= 10.0
        if nunique == total and df[col].dtype == "object":
            score -= 10.0  # Likely unique identifier

        scores[col] = score

    # Return best candidate if score > 0
    best_col = max(scores, key=scores.get)
    return best_col if scores[best_col] > 0 else None



















# ————————————————————————————————
# 🛠️ Tool 1: LoadData
# ————————————————————————————————

@tool
def LoadData(filepath: str) -> dict:
    """
    Loads data from a CSV file and returns it as a dictionary.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        dict: Data as dictionary (from DataFrame.to_dict()).
    """
    df = pd.read_csv(filepath)
    return df.to_dict()


# ————————————————————————————————
# 🛠️ Tool 2: CleanData (Enhanced)
# ————————————————————————————————

@tool
def CleanData(data: dict, handle_outliers: bool = True, impute_strategy: str = "median_mode") -> pd.DataFrame:
    """
    Cleans dataset with smart imputation, encoding, and optional outlier removal.

    Args:
        data (dict): Dataset in dictionary format.
        handle_outliers (bool): Whether to remove outliers using IQR.
        impute_strategy (str): "median_mode" or "mean_mode"

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = pd.DataFrame.from_dict(data)

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            if impute_strategy == "median_mode" or df[col].skew() > 1:
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
        else:
            mode = df[col].mode()
            fill_val = mode[0] if len(mode) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Parse datetime
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            except:
                pass

    # Encode categorical variables (only if not too many unique values)
    for col in df.select_dtypes(include="object").columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype("category").cat.codes
        # else: leave as object (e.g., free text)

    # Outlier removal (optional)
    if handle_outliers:
        for col in df.select_dtypes(include=["float64", "int64"]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            count_before = len(df)
            df = df[(df[col] >= lower) & (df[col] <= upper)]
            if len(df) == 0:
                # Avoid empty df
                df = pd.DataFrame.from_dict(data)  # Revert
                break

    return df.reset_index(drop=True)


# ————————————————————————————————
# 📊 Tool 3: EDA (Enhanced)
# ————————————————————————————————

@tool
def EDA(data: dict, max_cat_plots: int = 3, max_num_plots: int = 3) -> dict:
    """
    Performs advanced EDA with smart visualizations and insights.

    Args:
        data (dict): Dataset in dictionary format.
        max_cat_plots (int): Max number of categorical distribution plots.
        max_num_plots (int): Max number of numeric vs target plots.

    Returns:
        dict: EDA results including text, plots, and recommendations.
    """
    df = pd.DataFrame.from_dict(data)
    results = {}

    # 1. Summary Stats
    results["summary"] = df.describe(include="all").to_string()

    # 2. Missing Values
    missing = df.isnull().sum()
    results["missing_values"] = missing[missing > 0].to_dict()

    # Missingness heatmap
    if missing.sum() > 0:
        plt.figure(figsize=(8, 4))
        sns.heatmap(df.isnull(), cbar=True, cmap="viridis", yticklabels=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["missingness_plot"] = img #buf

    # 3. Correlation Heatmap
    corr = df.corr(numeric_only=True)
    if not corr.empty and len(corr.columns) > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["correlation_plot"] = img #buf

        # Top 5 absolute correlations
        unstacked = corr.abs().unstack()
        unstacked = unstacked[unstacked < 1.0]
        top_corr = unstacked.sort_values(ascending=False).head(5).to_dict()
        results["top_correlations"] = top_corr

    # 4. Skewness & Kurtosis
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    skew_kurt = {}
    for col in numeric_cols:
        skew_kurt[col] = {"skew": df[col].skew(), "kurtosis": df[col].kurtosis()}
    results["skew_kurtosis"] = skew_kurt

    # 5. Numeric Distributions
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(bins=20, figsize=(12, 8), layout=(2, -1))
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results["numeric_distributions"] = img #buf

    # 6. Categorical Distributions
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols[:max_cat_plots]:
        plt.figure(figsize=(6, 4))
        top_vals = df[col].value_counts().head(10)
        sns.barplot(x=top_vals.index, y=top_vals.values)
        plt.xticks(rotation=45)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        results[f"dist_{col}"] = img #buf

    # 7. Target Relationships
    target_col = detect_target_column(df)
    if target_col:
        results["detected_target"] = target_col
        for col in numeric_cols[:max_num_plots]:
            plt.figure(figsize=(6, 4))
            if df[target_col].nunique() <= 20:
                sns.boxplot(data=df, x=target_col, y=col)
            else:
                sns.scatterplot(data=df, x=col, y=target_col)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            plt.close()
            buf.seek(0)
            img = Image.open(buf)
            results[f"{col}_vs_{target_col}"] = img #buf

    # 8. Recommendations
    recs = []
    for col, sk in skew_kurt.items():
        if abs(sk["skew"]) > 1:
            recs.append(f"Feature '{col}' is skewed ({sk['skew']:.2f}) → consider log transform.")
    if results["missing_values"]:
        recs.append("Missing data detected → consider KNN or iterative imputation.")
    if results.get("top_correlations"):
        recs.append("High correlations found → consider PCA or feature selection.")
    if target_col:
        recs.append(f"Target variable '{target_col}' detected automatically.")
    results["recommendations"] = recs

    return results


# ————————————————————————————————
# 🤖 Tool 4: AutoML (Enhanced)
# ————————————————————————————————

@tool
def AutoML(data: dict, task_hint: str = None) -> dict:
    """
    Enhanced AutoML with multiple models and robust evaluation.

    Args:
        data (dict): Cleaned dataset.
        task_hint (str): "classification", "regression", or None.

    Returns:
        dict: Model results and metrics.
    """
    df = pd.DataFrame.from_dict(data)
    results = {}

    target_col = detect_target_column(df)
    if not target_col:
        results["note"] = "No target column detected. Check column names and data."
        return results

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # One-hot encode X
    X = pd.get_dummies(X, drop_first=True)

    if X.shape[1] == 0:
        results["error"] = "No valid features after encoding."
        return results

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Detect task
    if task_hint:
        task = task_hint
    elif y.dtype in ["object", "category"] or y.nunique() <= 20:
        task = "classification"
    else:
        task = "regression"

    try:
        if task == "classification":
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
            }
            results["task"] = "classification"
            best_acc = 0
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                if acc > best_acc:
                    best_acc = acc
                    results["accuracy"] = acc
                    results["best_model"] = name
                    results["report"] = classification_report(y_test, preds, zero_division=0)
                    if hasattr(model, "feature_importances_"):
                        results["feature_importance"] = dict(zip(X.columns, model.feature_importances_))

        else:
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "LinearRegression": LinearRegression()
            }
            results["task"] = "regression"
            best_r2 = -float("inf")
            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                if r2 > best_r2:
                    best_r2 = r2
                    results["r2_score"] = r2
                    results["mse"] = mean_squared_error(y_test, preds)
                    results["best_model"] = name
                    best_model = model  # Keep best model
                    if hasattr(model, "feature_importances_"):
                        results["feature_importance"] = dict(zip(X.columns, model.feature_importances_))
        # ✅ Save the best model to a temporary file
        model_dir = tempfile.mkdtemp()
        model_path = os.path.join(model_dir, f"trained_model_{task}.pkl")
        joblib.dump({
            "model": best_model,
            "task": task,
            "target_column": target_col,
            "features": X.columns.tolist()
        }, model_path)

        results["model_download_path"] = model_path
        results["model_info"] = f"Best model: {results['best_model']} | Task: {task} | Target: {target_col}"

    except Exception as e:
        results["error"] = f"Model training failed: {str(e)}"

    return results



model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=os.environ["HF_TOKEN"],
    provider="together",
    max_tokens=8048
)

planner = ToolCallingAgent(
    tools=[],
    model=model,
    name="PlannerAgent",
    max_steps=10,
    planning_interval=5,
    description= "Breaks down complex tasks and orchestrates tools for execution",
)

# Research agent
researcher = ToolCallingAgent(
    tools=[internet_search],
    model=model,
    name="ResearchAgent",
    max_steps=10,
    description = "Conducts deep research using internet_search",
)

# Coding agent
coder = CodeAgent(
    tools=[
        code_executor,
        save_files,
        list_workspace_files,
        package_artifact,
        cleanup_workspace,
        run_manifest,
    ],
    model=model,
    name="CodingAgent",
    max_steps=20,
    additional_authorized_imports=[
        "subprocess", "tempfile", "time", "os", "pathlib", "typing","shutil", "zipfile","uuid"
    ],
    description = "Executes Python code safely in a sandboxed Docker container."
                  "If a library is missing, add it to install_libs in run_manifest."
)


analyst = CodeAgent(
    tools=[LoadData, CleanData, EDA, AutoML],
    model=model,
    max_steps=20,
    name="DataScienceAgent",
    additional_authorized_imports=[
        "pandas", "matplotlib.pyplot", "seaborn", "PIL", "sklearn", "io", "os","joblib","tempfile"
    ],
    description = "Loads datasets, cleans and preprocesses data, performs exploratory data analysis (EDA) with visualizations, and builds predictive models when a target variable is specified."
)


manager_agent = ToolCallingAgent(
    tools=[],
    model=model,
    managed_agents=[planner, researcher, coder, analyst],
    max_steps=20,
    description= "Routes user queries to the right agent (Planner, Researcher, Coder or Data Scientist) and assembles results",
)



# ------------
# -  GRADIO  -
# ------------


#def run_agent(user_input):

    #return manager_agent.run(user_input)





