---
trigger: always_on
---

When using running makefiles, you can pass the -j${n_proc} argument to parallelise compilation.
Adding this (with e.g. -j=8) is usually a good idea to speed up work.