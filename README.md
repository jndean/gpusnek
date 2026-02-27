# GPUSnek

A full python interpreter running on the GPU.
Even the parser and bytecode compiler.


Takes the src code from micropython, rams it through nvcc, apologises for nothing.


```bash
cd ports/cuda
make TARGET=cuda
./build-cuda/micropython
```

You can also build for the `TARGET=host`, useful for checking you haven't broken anything :)