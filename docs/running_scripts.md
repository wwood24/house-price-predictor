# Runing Scripts within vs

Since scripts are in in a script folder this sets the path to just the directory and it will search down but now anywhere else.  having .env and settings and launch settings to set the path to src is all skiped when clicking the run button except through debug mode that will work.  The only other way is from terminal with this command

```python
PYTHONPATH=src python scripts/script_file.py
```