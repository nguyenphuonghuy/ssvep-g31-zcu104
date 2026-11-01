# Vitis HLS 2024.2, kernel v2: G3_T=100|250, G3_STEPS="csim csynth cosim export"
set T $::env(G3_T)
set steps $::env(G3_STEPS)
set vec [file normalize vectors_rt/T$T]
set fresh [expr {[lsearch $steps csynth] >= 0}]
if {$fresh} { open_project -reset prj_rt_T$T } else { open_project prj_rt_T$T }
set_top g3_decode_rt
add_files src_rt/g3_decode_rt.cpp -cflags "-I$vec -std=c++14"
add_files -tb src_rt/tb_g3_rt.cpp -cflags "-I$vec -I[file normalize src_rt] -std=c++14"
if {$fresh} { open_solution -reset sol -flow_target vivado } else { open_solution sol -flow_target vivado }
set_part xczu7ev-ffvc1156-2-e
create_clock -period 5 -name default
config_export -format ip_catalog -rtl verilog -ipname g3_decode_rt_T$T
if {[lsearch $steps csim] >= 0}   { csim_design -argv "$vec 1,2 8 $vec/expected_phi.bin" }
if {[lsearch $steps csynth] >= 0} { csynth_design }
if {[lsearch $steps cosim] >= 0}  { cosim_design -argv "$vec 1,2 8" -rtl verilog }
if {[lsearch $steps export] >= 0} { export_design -format ip_catalog }
exit
