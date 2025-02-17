#!/bin/bash

# execute with
# ./test_remap.sh 4      -- cpu

file="autotest/out_remap.dat"

ntask=$1

command="mpirun -np "$((ntask))" ./remhos -no-vis -dt -1.0"

methods=( "-opt 0"      # no optimization
          "-opt 1"      # HiOp
          "-opt 1 -h1s" # HiOp with h1-seminorm
          "-opt 2")     # LVPP

cd ..
make
rm -f $file

for method in "${methods[@]}"; do

  echo -e '\n'"--- Method "$method >> $file

  echo -e '\n'"- 2D scalar pacman GridFunction" >> $file
  run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 4 -tf 0.75 -mono 3 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Mass initial' -e 'Mass interpolated diff:' -e 'Mass optimized diff:' >> $file

  echo -e '\n'"- 2D scalar analytic smooth GridFunction" >> $file
  run_line=$command" -m ./data/inline-quad.mesh -p 13 -rs 2 -tf 0.75 -mono 3 -proj "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Mass initial' -e 'Mass interpolated diff:' \
                   -e 'Mass optimized diff:' -e 'L1 error:' >> $file

  echo -e '\n'"- 3D scalar GridFunction" >> $file
  run_line=$command" -m ./data/cube01_hex.mesh -p 10 -rs 3 -o 2 -tf 0.5 -mono 3 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Mass initial' -e 'Mass interpolated diff:' -e 'Mass optimized diff:' >> $file

  echo -e '\n'"- 2D scalar pacman QuadratureFunction" >> $file
  run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 4 -tf 0.75 -mono 4 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Mass initial' -e 'Mass interpolated diff:' -e 'Mass optimized diff:' >> $file

  echo -e '\n'"- 3D scalar QuadratureFunction" >> $file
  run_line=$command" -m ./data/cube01_hex.mesh -p 10 -rs 3 -o 2 -tf 0.5 -mono 4 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Mass initial' -e 'Mass interpolated diff:' -e 'Mass optimized diff:' >> $file

  echo -e '\n'"- 2D ind_rho_e" >> $file
  run_line=$command" -m ./data/inline-quad.mesh -p 14 -rs 4 -tf 0.75 -mono 5 "$method
  echo -e $run_line >> $file
  $run_line | grep -e 'Volume interpolated diff:' -e 'Mass interpolated diff:' -e 'Energy interpolated diff:'>> $file

done

cd autotest

tkdiff out_baseline_remap.dat out_remap.dat &

exit 0
