# GPUSnek

A full python interpreter running on the GPU.
Even the parser and bytecode compiler.
For no good reason.


```bash
cd ports/cuda
make TARGET=cuda
./build-cuda/micropython
```

You can also build for the `TARGET=host`, useful for checking you haven't broken anything :)