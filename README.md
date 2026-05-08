<p align="center">
  <img src="logo.png" width=500px>
</p>


Read the ["whitepaper"](https://josefdean.co.uk/gpusnek.pdf) here.

**gpusnek** answers the question "What would it look like to be able to inline arbitrary Python code into your high-performance CUDA kernels, with no consideration for why that is a bad idea?".

This repository implements a full Python interpreter that can run on one GPU thread (or in parallel on many).
It even includes the Python lexer, parser and bytecode compiler.

We take the source code from MicroPython, ram it through _nvcc_ (NVIDIA's CUDA compiler), and fix most of the things which break.

Examples include:
 - Running 1 Million Python interpreters on a consumer GPU and using them in an interactive REPL.
 - Communicating between CUDA threads by using Python to read/write to a shared virtual filesystem living in VRAM
 - Other such nonsense.

```bash
# Assuming you have CUDA development tools set up
make TARGET=cuda -j
./example_allreduce
```

You can also build for the `TARGET=host`, useful for checking you haven't broken anything :)
