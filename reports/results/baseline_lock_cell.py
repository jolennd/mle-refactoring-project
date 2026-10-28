import sys, nbformat
path, sha = sys.argv[1], sys.argv[2]
nb = nbformat.read(path, as_version=4)
src = f'''# Appended for the baseline lock (ADR-006); not part of the frozen notebook.
import json, datetime, sklearn
lock = {{
    "cv_r2": float(elastic_cv.best_score_),
    "best_params": {{k: v for k, v in elastic_cv.best_params_.items()}},
    "test_r2": float(elastic.score(X_test_poly, y_test)),
    "source_notebook": "King-County.ipynb",
    "source_notebook_sha256": "{sha}",
    "sklearn_version": sklearn.__version__,
    "created": datetime.date.today().isoformat(),
    "procedure": "nbconvert --execute of an unmodified copy of King-County.ipynb plus this appended cell",
}}
with open("baseline_lock.json", "w") as f:
    json.dump(lock, f, indent=2)
    f.write("\\n")
print(json.dumps(lock, indent=2))
'''
nb.cells.append(nbformat.v4.new_code_cell(src))
nbformat.write(nb, path)
