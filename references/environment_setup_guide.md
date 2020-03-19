# Environment set up notes
1. To set up the environment yaml file, the basic steps are
   1. Create a new conda environment.
   2. Install the desired packages.
   3. Create a yaml file listing the packages in the environment.
   Be sure to use the command `conda env export --no-builds -f environment.yml` so one doesn't export the packages in a format that cannot be used by Binder.
   4. Clean the yaml file by removing operating-system specific packages.
   This allows the repo to launched on Binder.
   The likely packages to be removed are:
      1. libcxx=4.0.1
      2. libcxxabi=4.0.1
      3. libgfortran=3.0.1
      4. llvm-openmp=4.0.1
