# Vitis HLS 2024.2: vitis_hls -f run_hls.tcl  (bien moi truong G3_T=100|250, G3_STEPS="csim csynth cosim export")
set T $::env(G3_T)
set steps $::env(G3_STEPS)
set vec [file normalize vectors/T$T]
set fresh [expr {[lsearch $steps csynth] >= 0}]
if {$fresh} { open_project -reset prj_T$T } else { open_project prj_T$T }
set_top g3_decode
add_files src/g3_decode.cpp -cflags "-I$vec -std=c++14"
add_files -tb src/tb_g3.cpp -cflags "-I$vec -I[file normalize src] -std=c++14"
if {$fresh} { open_solution -reset sol -flow_target vivado } else { open_solution sol -flow_target vivado }
set_part xczu7ev-ffvc1156-2-e
create_clock -period 5 -name default
config_export -format ip_catalog -rtl verilog -ipname g3_decode_T$T
if {[lsearch $steps csim] >= 0}   { csim_design -argv "$vec 8" }
if {[lsearch $steps csynth] >= 0} { csynth_design }
if {[lsearch $steps cosim] >= 0}  { cosim_design -argv "$vec 8 nophi" -rtl verilog }
if {[lsearch $steps export] >= 0} { export_design -format ip_catalog }
exit
