# Vivado 2024.2: vivado -mode batch -source reports.tcl -tclargs <tag, vd g3_T100>  (sau build_overlay.tcl). Bao cao sau route.
set_param general.maxThreads 2
set root [file normalize .]
set tag [lindex $argv 0]
open_checkpoint [glob $root/vivado_$tag/g3_ov.runs/impl_1/g3_wrapper_routed.dcp]
file mkdir $root/reports_$tag
report_utilization -file $root/reports_$tag/util_flat.rpt
report_utilization -hierarchical -hierarchical_depth 3 -file $root/reports_$tag/util_hier.rpt
report_timing_summary -file $root/reports_$tag/timing_summary.rpt
# cong suat uoc tinh (vectorless: ty le chuyen mach mac dinh cua Vivado; khong co SAIF)
report_power -file $root/reports_$tag/power.rpt -hierarchical_depth 3
puts "CLK: [get_property PERIOD [get_clocks clk_pl_0]] ns"
puts "WNS = [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
puts "WHS = [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -hold]]"
puts "REPORTS_DONE"
