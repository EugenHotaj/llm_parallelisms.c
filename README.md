# LLM parallelisms in C, spelled out

`parallelisms` is an educational library which implements SOTA LLM parallelisms in pure C. 
It aims to have a simple, spelled out implementations for maximum clarity. Currently supports:
  - Data parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/train_dp.c)]
  - Fully-sharded data parallel (FSDP) [[code](https://github.com/EugenHotaj/ml.c/blob/main/train_fsdp.c)]
  - Tensor parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/train_tp.c)]
  - Pipeline parallel [[code](https://github.com/EugenHotaj/ml.c/blob/main/train_pp.c)]
  - **Advanced 3D Parallelism**: combining FSDP, tensor, and pipeline parallelism [[code](https://github.com/EugenHotaj/ml.c/blob/main/train_3d.c)]


While everything runs on CPU using MPI, the key ideas illustrated here are exactly how SOTA LLMs (like [Llama 3](https://arxiv.org/abs/2407.21783)) are trained.

## Getting started

The only dependency is OpenMPI which can be downloaded [here](https://docs.open-mpi.org/en/v5.0.x/quickstart.html) (but any MPI implementation will work). Once you've installed 
OpenMPI, you can compile and run any of the training scripts, for example `train_fsdp.c`:

```
mpicc -Ofast train_fsdp.c && mpirun -n 4 a.out
```

Here `-n` specifies the number of FSDP shards. If you don't have enough cores for the desired parallelism level, you can tell OpenMPI to oversubscribe the cores. For example, here
is how I run 3d parallelism on my 8 core MacBook Air:

```
mpicc -Ofast train_3d.c && mpirun -n 24 --map-by=:oversubscribe a.out --tp 4 --dp 2
```

This will tell OpenMPI to run with 24 shards and the training script will use tensor parallelism of 4, (fully sharded) data parallelism of 2, and pipeline parallelism of 3 (this is always fixed).

The training scripts will all train a character-level language model on a names dataset (similar to [makemore](https://github.com/karpathy/makemore/tree/master)). At the end of training, 10 new names
are generated, for example:

```
Final validation loss: 2.343800
. . . . . . . . . . . . a n s i  --> .
. . . . . . . . f r e s s i n i  --> .
. . . . . . . . . . m a y d r a  --> .
. . . . . . . . . . m e r g i n  --> .
. . . . . . . . p l a y o n n g  --> .
. . . . . . . . . . j e n i c z  --> .
. . . . . . . . . . . . e a i s  --> .
. . . . . . . . . . . m e r m y  --> .
. . . . . . . . . m i l e s e n  --> .
. . . . . . . . . . s h a b e r  --> .
```

All randomness is seeded and the models are initialized identically across all training scripts so the generations should all be identical in theory. In practice however, the 
trained models do diverge slightly and produce slightly different generations. This is likley due to the non-commutativity of floating point operations arising from MPI messages
arriving in different orders.

## Technical details for the curious

The most interesting files to look at are the `train_*.c` files. `train.c` contains the reference implementation for single-threaded training
while the rest implement the individual parallelism methods. Finally, `train_3d.c` brings everything together to implement 3d parallel training. 
The individual parallelism implementations are modular enough that 3d parallelism "falls out".

The rest of the code implements an MLP with hand-crafted forward, backward passes and a data loader.

* The forward/backward functions have the exact same signature. This makes calling the backwards pass convenient as it's just a mirrored version of the forward pass.
* C only supports `structs` and `functions` (no classes) so "methods" are implemented by prefixing functions by the name of the struct they "belong" and passing the struct as `self`.
* Design tradeoffs were made to favor clarity over performance. For example, each script creates the full model in memory before sharding it to ensure initialization is identical across scripts. This is wasteful in practice and requires much more memoryh than is necessary.
