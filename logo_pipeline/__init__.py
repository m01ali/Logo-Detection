"""M3 logo-detection pipeline (Revision 3, T7-T11 foundation).

One YAML drives a run of discrete, persisted stages. See ``spec docs/03`` for
the architecture and ``spec docs/07`` for the MM0/MM1 build this package
implements. Heavy dependencies (torch, transformers, easyocr, ultralytics)
are imported inside the stages that need them so the CPU path runs on a
machine without them.
"""

__version__ = "0.1.0"
