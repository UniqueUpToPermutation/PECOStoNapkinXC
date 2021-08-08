# Performance Comparison between PECOS and NapkinXC

This repository was created to have a head-to-head comparison between [PECOS](https://github.com/amzn/pecos) and [NapkinXC](https://github.com/mwydmuch/napkinXC). They are both extreme multi-label ranking (XMR) models.

I've had to make small modifications to both to enable model conversion; these tweaks do not affect performance. They only changes which were made:

1. CMakeLists in NapkinXC was changed to build NapkinXC as a static library.
2. A number of classes in NapkinXC now have all of their fields exposed as public.
3. I've removed OpenMP from PECOS.

This code is provided for performance diagnostics only. **You should not use it in any production projects**.

## To convert models from PECOS to NapkinXC

To convert models from PECOS to NapkinXC build the CMake target ModelConv and call:

```
ModelConv [model_path_1] [model_path_2] ... [model_path_n]
```

The **model_i** arguments are the paths to the rankers. ModelConv expects there to be a **ranker** folder in each of the paths **model_i**. The resulting NapkinXC models will be written to **[model_path_i]/../napkin-model**.

If no arguments are provided, ModelConv will search the **./data/** folder relative to the CMake project root.

**Note**: while I've been able to load the models into NapkinXC via C++, I haven't been able to load them in via the NapkinXC Python interface.

## To benchmark PECOS and NapkinXC inference times

To benchmark PECOS models against NapkinXC models build the CMake target ModelBenchmark and call:

```
ModelBenchmark [dataset_path_1] [dataset_path_2] ... [dataset_path_n]
```

where **[dataset_path_i]** contains a **X.tst.tfidf.npz** (test data features) file and a **Y.tst.npz** file (test data matches), as well as a **model** folder containing a PECOS model and a **napkin-model** folder containing a NapkinXC model. ModelBenchmark will load test data and models for each of the datasets and spit out precision/recall metrics as well as CPU time on one thread.

If no arguments are provided, ModelBenchmark will benchmark all subdirectories of the **./data/** folder relative to the CMake project root.