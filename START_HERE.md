# Start here (fresh start)

`PLAN.md` explains the stages and what to do with the results. This file is the exact first steps.

## 1. Set up the Mac (about 10 min)
```bash
cd ~/Downloads
unzip -o cs228-pad-project.zip && cd cs228-pad-project
bash scripts/setup_mac.sh
source venv/bin/activate
```
You want to see `Apple GPU (MPS) available: True`.

## 2. Prove everything works (about 10 min)
```bash
./scripts/smoke_test.sh
```
It runs the real pipeline on fake data (dataset prep, baseline, 3 hardened variants, evaluation,
report, bundle, ONNX export, attacked demo images) and ends with `SMOKE TEST PASSED`.

## 3. Put the code on GitHub (Kaggle clones it from there)
If you deleted the old GitHub repo too:
```bash
git init && git add . && git commit -m "pad project"
git branch -M main
gh repo create cs228-pad-project --public --source=. --push
```
If the repo `ShivamShelar101/cs228-pad-project` still exists (it holds the OLD code), overwrite it:
```bash
git init && git add . && git commit -m "pad project"
git branch -M main
git remote add origin https://github.com/ShivamShelar101/cs228-pad-project.git
git push -u --force origin main
```
Check: open https://github.com/ShivamShelar101/cs228-pad-project and look for `scripts/run_experiment.sh`
(the old code had `run_crunch.sh`).

## 4. Run the experiment on Kaggle
1. kaggle.com -> Create -> New Notebook. File -> Import Notebook -> upload `notebooks/kaggle_run.ipynb`.
2. Session options: **GPU on, Internet on**. (You do not need to add the dataset yourself; cell 2 downloads it.)
3. Run the cells one at a time, reading each output. Details are in the notebook.
4. Download `bundle.zip` from the Output panel, then stop the session (power icon).

## 5. Demo and report
See `PLAN.md`.
