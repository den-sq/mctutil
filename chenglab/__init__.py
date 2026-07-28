"""Cheng-Lab specific adapters for mctutil generic tools.

The contents of this package encode conventions (folder layouts, status enums,
sbatch parameter formats, Google Sheets layouts) that are specific to the
Cheng Lab's micro-CT pipeline. They are kept in-repo as the default schema
for `mctutil parse meta-shift` but live behind the generic adapter seam in
parsing/meta_shift.py so additional schemas can be added without touching the
engine.
"""
