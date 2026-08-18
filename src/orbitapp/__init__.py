"""orbitapp: a PySide6 GUI for the orbit assessment-forward imaging pipeline.

Launch via ``python3 -m orbitapp`` (or the ``orbitapp`` console script).
One tab per pipeline stage (see docs/nph.pdf Fig. 3 for the stage order);
each tab only calls functions from ``orbit`` and displays their results --
it never computes anything itself.
"""

__version__ = "0.1.0"
