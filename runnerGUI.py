#!/usr/bin/env python3
"""
Unified ML Runner (Lisp / C / Java)
CMPS 3500 - Final Project Framework

Features:
- Dataset + target selection with validation
- Pre-screen: choose implementation (C / Java / Lisp / General results)
- Per-implementation algorithm menu
- Timing per language
- Automatic SLOC scanning
- Robust Lisp error handling
- Safe general results table
- Unified CSV output
"""

import os
import csv
import subprocess
import time

try:
    import resource
except ImportError:
    resource = None


# -----------------------------------------------------------
# PATHS & DEFAULTS
# -----------------------------------------------------------

BASE = os.path.dirname(os.path.abspath(__file__))
FP_DIR = os.path.join(BASE, "fp")
PROC_DIR = os.path.join(BASE, "proc")
JAVA_DIR = os.path.join(BASE, "oop-java")
RESULTS_DIR = os.path.join(BASE, "results")

LISP_SCRIPT = os.path.join(FP_DIR, "Lisp-Algorithm")

DEFAULT_DATA = os.path.join(BASE, "data", "adult_income_cleaned.csv")
DEFAULT_TARGET = "income"

USER_DATA = DEFAULT_DATA
USER_TARGET = DEFAULT_TARGET

C_RESULTS = os.path.join(PROC_DIR, "c_model_results.csv")
JAVA_RESULTS = os.path.join(RESULTS_DIR, "java_results.csv")
UNIFIED = os.path.join(RESULTS_DIR, "unified_results.csv")


# -----------------------------------------------------------
# MODEL CONFIG
# -----------------------------------------------------------

ALGS = {
    "knn": {
        "label": "KNN",
        "task": "classification",
        "lisp_csv": "knn_results.csv",
        "c_name": "K-Nearest Neighbors (k=7)",
        "java_key": "knn",
    },
    "logistic": {
        "label": "Logistic Regression",
        "task": "classification",
        "lisp_csv": "logistic_regression_results.csv",
        "c_name": "Logistic Regression",
        "java_key": "logistic",
    },
    "naive_bayes": {
        "label": "Gaussian Naive Bayes",
        "task": "classification",
        "lisp_csv": "naive_bayes_results.csv",
        "c_name": "Gaussian Naive Bayes",
        "java_key": "naivebayes",
    },
    "decision_tree": {
        "label": "Decision Tree (ID3)",
        "task": "classification",
        "lisp_csv": "decision_tree_results.csv",
        "c_name": "Decision Tree (ID3)",
        "java_key": "tree",
    },
    "linear_regression": {
        "label": "Linear Regression",
        "task": "regression",
        "lisp_csv": "linear_regression_results.csv",
        "c_name": "Linear Regression",
        "java_key": "linear",
    },
}


# -----------------------------------------------------------
# SLOC SCANNING
# -----------------------------------------------------------

def list_files_recursive(folder, extensions):
    collected = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(ext) for ext in extensions):
                collected.append(os.path.join(root, f))
    return collected

C_SOURCE_FILES = list_files_recursive(PROC_DIR, [".c", ".h"])
JAVA_SOURCE_FILES = list_files_recursive(JAVA_DIR, [".java"])
LISP_SOURCE_FILES = [LISP_SCRIPT]

SLOC_FILES = {
    "C":   {alg: C_SOURCE_FILES for alg in ALGS},
    "Java":{alg: JAVA_SOURCE_FILES for alg in ALGS},
    "Lisp":{alg: LISP_SOURCE_FILES for alg in ALGS},
}

def count_sloc_file(path):
    count = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if (s.startswith("//") or s.startswith("/*")
                    or s.startswith("*") or s.startswith("#")
                    or s.startswith(";")):
                    continue
                count += 1
    except:
        return 0
    return count

def get_sloc(lang, alg_key):
    return sum(count_sloc_file(f) for f in SLOC_FILES.get(lang, {}).get(alg_key, []))


# -----------------------------------------------------------
# CSV VALIDATION
# -----------------------------------------------------------

def validate_csv_columns(csv_path, target_col):
    if not os.path.exists(csv_path):
        print(f"\n[ERROR] CSV file not found:\n  {csv_path}\n")
        return False, []

    try:
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip().split(",")
    except Exception as e:
        print(f"\n[ERROR] Unable to read CSV file:\n  {csv_path}\n{e}\n")
        return False, []

    header = [h.strip().replace('"', "") for h in header]

    if target_col not in header:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("[ERROR] Target column NOT FOUND in dataset!")
        print(f"Missing: '{target_col}'")
        print("Available columns:\n  " + ", ".join(header))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        return False, header

    return True, header


# -----------------------------------------------------------
# PRINT UTILITIES
# -----------------------------------------------------------

def print_table(headers, rows):
    if not rows:
        print("(no data)\n")
        return

    widths = [max(len(str(r[i])) for r in [headers] + rows)
              for i in range(len(headers))]

    border = "+" + "+".join("-"*(w+2) for w in widths) + "+"

    def fmt(row):
        return "|" + "|".join(
            " "+str(row[i]).ljust(widths[i])+" " for i in range(len(row))
        ) + "|"

    print(border)
    print(fmt(headers))
    print(border)
    for r in rows:
        print(fmt(r))
    print(border)
    print()

def measure_time(fn, *args):
    start = time.perf_counter()
    fn(*args)
    return time.perf_counter() - start


# -----------------------------------------------------------
# LISP PIPELINE
# -----------------------------------------------------------

def run_lisp(alg):
    target = USER_TARGET
    extra_args = ""

    if alg == "linear_regression":
        target = "hours.per.week"
        extra_args = " :l2 1.0"

    form = {
        "knn": f'(ml-framework:run-knn "{USER_DATA}" "{target}")',
        "logistic": f'(ml-framework:run-logistic-regression "{USER_DATA}" "{target}")',
        "naive_bayes": f'(ml-framework:run-naive-bayes "{USER_DATA}" "{target}")',
        "decision_tree": f'(ml-framework:run-decision-tree "{USER_DATA}" "{target}")',
        "linear_regression":
            f'(ml-framework:run-linear-regression "{USER_DATA}" "{target}"{extra_args})'
    }[alg]

    try:
        result = subprocess.run(
            ["sbcl","--noinform","--disable-debugger",
             "--load",LISP_SCRIPT,"--eval",form,"--quit"],
            cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode != 0:
            print(f"\n[LISP WARNING] SBCL returned code {result.returncode}")
            print("LISP STDERR (trimmed):")
            print(result.stderr[:300] if result.stderr else "(none)")
            print()

    except Exception as e:
        print("\n[LISP ERROR] failed to run.")
        print(e)


def read_lisp_metrics(file, task):
    path = os.path.join(BASE, file)
    if not os.path.exists(path): return None
    rows = list(csv.DictReader(open(path)))
    if not rows: return None
    last = rows[-1]
    if task == "classification":
        return {
            "metric1": last.get("Accuracy",""),
            "metric2": last.get("Macro-F1",""),
        }
    return {
        "metric1": last.get("RMSE",""),
        "metric2": last.get("R^2",""),
    }


# -----------------------------------------------------------
# C PIPELINE
# -----------------------------------------------------------

def raise_stack():
    if resource:
        try:
            resource.setrlimit(resource.RLIMIT_STACK,
                               (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
        except:
            pass

def build_c_csv():
    out = os.path.join(PROC_DIR,"temp_c_input.csv")
    with open(USER_DATA,"r",encoding="utf-8",errors="ignore") as fin, \
         open(out,"w",encoding="utf-8") as fout:

        fout.write(fin.readline())
        for line in fin:
            cells = [c.replace(",", " ").replace('"', "")[:120]
                     for c in line.strip().split(",")]
            fout.write(",".join(cells) + "\n")

    return "temp_c_input.csv"

def compile_c():
    try:
        subprocess.run(["make"], cwd=PROC_DIR, check=True)
    except:
        print("[C] Make error")

def run_c():
    raise_stack()
    safe = build_c_csv()
    try:
        subprocess.run(
            ["./ml_program", safe, USER_TARGET, "0.3"],
            cwd=PROC_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        print("[C] ERROR")

def read_c_metrics(name):
    if not os.path.exists(C_RESULTS): return None
    for r in csv.DictReader(open(C_RESULTS)):
        if r.get("Model") == name:
            return {
                "metric1": r.get("Metric1_Value",""),
                "metric2": r.get("Metric2_Value","")
            }
    return None


# -----------------------------------------------------------
# JAVA PIPELINE
# -----------------------------------------------------------

def compile_java():
    build = os.path.join(JAVA_DIR,"build")
    os.makedirs(build, exist_ok=True)
    try:
        subprocess.run([
            "javac","-d","build",
            "ml/models/LinearRegression.java",
            "ml/models/LogisticRegression.java",
            "ml/models/KNearestNeighbors.java",
            "ml/models/DecisionTree.java",
            "ml/models/GaussianNaiveBayes.java",
            "ml/models/BaseModel.java",
            "ml/models/Model.java",
            "ml/models/ModelMetrics.java",
            "ml/data/DataLoader.java",
            "ml/data/Dataset.java",
            "ml/utils/MetricsUtil.java",
            "MLLibraryApp.java"
        ], cwd=JAVA_DIR, check=True)
    except:
        print("[JAVA] Compile error")

def run_java(algo, label):
    try:
        if os.path.exists(JAVA_RESULTS):
            os.remove(JAVA_RESULTS)
    except: pass

    try:
        subprocess.run(
            ["java","-cp","build","MLLibraryApp",
             "--batch","--algorithm",algo,"--data",USER_DATA],
            cwd=JAVA_DIR, capture_output=True, text=True, check=True)
    except:
        print("[JAVA] ERROR")

def read_java_metrics(key, task):
    if not os.path.exists(JAVA_RESULTS): return None
    rows = [r for r in csv.DictReader(open(JAVA_RESULTS))
            if r.get("Algorithm") == key]
    if not rows: return None
    last = rows[-1]
    return {"metric1": last.get("Metric1",""),
            "metric2": last.get("Metric2","")}


# -----------------------------------------------------------
# UNIFIED RESULTS
# -----------------------------------------------------------

def save_unified(alg_key, cfg, lm, cm, jm):
    fresh = not os.path.exists(UNIFIED)
    with open(UNIFIED,"a",newline="") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["Algorithm","Task","Language","Metric1","Metric2","Time","SLOC"])

        def wr(lang, m):
            if not m: return
            t = m.get("time","")
            if isinstance(t,(int,float)): t = f"{t:.6f}"
            sloc = get_sloc(lang, alg_key)
            w.writerow([cfg["label"], cfg["task"], lang,
                        m["metric1"], m["metric2"], t, sloc])

        wr("Lisp", lm)
        wr("C", cm)
        wr("Java", jm)


# -----------------------------------------------------------
# RUN ALGORITHM
# -----------------------------------------------------------

def run_algorithm(alg_key, impl=None):
    cfg = ALGS[alg_key]

    print("\n========================================")
    print(f"Running {cfg['label']}")
    print(f"CSV:    {USER_DATA}")
    print(f"Target: {USER_TARGET}")
    print("========================================\n")

    lm = cm = jm = None

    if impl in (None,"Lisp"):
        t = measure_time(run_lisp, alg_key)
        lm = read_lisp_metrics(cfg["lisp_csv"], cfg["task"])
        if lm: lm["time"] = t

    if impl in (None,"C"):
        compile_c()
        t = measure_time(run_c)
        cm = read_c_metrics(cfg["c_name"])
        if cm: cm["time"] = t

    if impl in (None,"Java"):
        compile_java()
        t = measure_time(run_java, cfg["java_key"], cfg["label"])
        jm = read_java_metrics(cfg["java_key"], cfg["task"])
        if jm: jm["time"] = t

    headers = ["Language","Accuracy","Macro-F1","Time","SLOC"] \
              if cfg["task"]=="classification" else \
              ["Language","RMSE","R^2","Time","SLOC"]

    rows=[]
    def add(lang, m):
        if not m:
            rows.append([lang,"ERROR","ERROR","ERROR","N/A"])
            return
        t = m.get("time","")
        if isinstance(t,(int,float)): t = f"{t:.6f}"
        sloc = get_sloc(lang, alg_key)
        rows.append([lang,m["metric1"],m["metric2"],t,sloc])

    if impl in (None,"Lisp"): add("Lisp", lm)
    if impl in (None,"C"):    add("C", cm)
    if impl in (None,"Java"): add("Java", jm)

    print_table(headers, rows)
    save_unified(alg_key, cfg, lm, cm, jm)


# -----------------------------------------------------------
# GENERAL RESULTS
# -----------------------------------------------------------

def print_general_results(filter_impl=None):
    if not os.path.exists(UNIFIED) or os.path.getsize(UNIFIED)==0:
        print("\nNo unified results yet.\n")
        return

    data = list(csv.DictReader(open(UNIFIED)))
    if not data:
        print("\nNo unified results yet.\n")
        return

    rows=[]
    for r in data:
        lang = r["Language"]
        if filter_impl and lang != filter_impl:
            continue
        rows.append([
            lang,
            r["Algorithm"],
            r["Time"],
            r["Metric1"],
            r["Metric2"],
            r["SLOC"],
        ])

    print("\nGeneral Results (Comparison)")
    print("***********************************************\n")
    print_table(["Impl","Algorithm","Time","Metric1","Metric2","SLOC"], rows)


# -----------------------------------------------------------
# SETUP USER OPTIONS
# -----------------------------------------------------------

def setup_user_options():
    global USER_DATA, USER_TARGET
    
    while True:
        print("\n========================================")
        print("Dataset & Target Configuration")
        print("========================================")
        print(f"Default CSV: {DEFAULT_DATA}")
        
        # Get CSV
        x = input("Enter CSV (Enter = default): ").strip()
        
        # Convert to ABSOLUTE path immediately
        if x:
            USER_DATA = os.path.abspath(x)
        else:
            USER_DATA = DEFAULT_DATA

        print(f"\nDefault target: {DEFAULT_TARGET}")
        t = input("Enter target (Enter = default): ").strip()
        USER_TARGET = t if t else DEFAULT_TARGET

        # NOW validate the actual absolute path (CRITICAL FIX)
        valid, cols = validate_csv_columns(USER_DATA, USER_TARGET)

        if valid:
            print(f"\nUsing CSV:    {USER_DATA}")
            print(f"Using Target: {USER_TARGET}\n")
            break

        print("Choose an option:")
        print("  (1) Enter NEW target")
        print("  (2) Choose NEW dataset")
        print("  (3) Quit")
        opt = input("Enter choice: ").strip()

        if opt == "1":
            print("\nAvailable columns:\n  " + ", ".join(cols))
            USER_TARGET = input("Enter new target: ").strip()
        elif opt == "2":
            continue
        else:
            exit(0)


# -----------------------------------------------------------
# IMPLEMENTATION MENU
# -----------------------------------------------------------

ALG_MENU = {
    "2":"linear_regression",
    "3":"logistic",
    "4":"knn",
    "5":"decision_tree",
    "6":"naive_bayes",
}

def implementation_menu(impl):
    while True:
        print("\n***********************************************")
        print(f"You selected: {impl}")
        print("***********************************************")
        print("(1) Load data (no-op)")
        print("(2) Linear Regression")
        print("(3) Logistic Regression")
        print("(4) KNN")
        print("(5) Decision Tree")
        print("(6) Gaussian Naive Bayes")
        print("(7) Print results")
        print("(8) Quit to main menu\n")

        c = input("Enter choice: ").strip()

        if c == "1":
            print("\nData loads automatically.\n")

        elif c in ALG_MENU:
            run_algorithm(ALG_MENU[c], impl=impl)

        elif c == "7":
            print_general_results(filter_impl=impl)

        elif c == "8":
            return

        else:
            print("Invalid option.\n")


# -----------------------------------------------------------
# MAIN MENU
# -----------------------------------------------------------

def main():
    setup_user_options()

    while True:
        print("\n***********************************************")
        print("Welcome to the AI/ML Library Implementation Comparison")
        print("***********************************************")
        print("(1) Procedural (C)")
        print("(2) Object-Oriented (Java)")
        print("(3) Functional (Lisp)")
        print("(4) Print General Results")
        print("(5) Quit\n")

        c = input("Enter choice: ").strip()

        if c == "1":
            implementation_menu("C")
        elif c == "2":
            implementation_menu("Java")
        elif c == "3":
            implementation_menu("Lisp")
        elif c == "4":
            print_general_results()
        elif c == "5":
            print("Exiting...")
            return
        else:
            print("Invalid option.\n")


if __name__ == "__main__":
    main()

