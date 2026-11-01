# Vivado 2024.2: vivado -mode batch -source build_overlay.tcl -tclargs <PL_MHZ> "<danh sach T, vd 100 hoac 100 250>"
# Overlay PYNQ: Zynq US+ (preset ZCU104) + g3_decode_T100 + g3_decode_T250; dieu khien HPM0, du lieu HP0.
set mhz [lindex $argv 0]
set Ts [lindex $argv 1]
set rt [expr {[llength $argv] > 2 && [lindex $argv 2] eq "rt"}]
set pfx [expr {$rt ? "rt_" : ""}]
set tag g3[expr {$rt ? "rt" : ""}]_T[join $Ts _]
set_param general.maxThreads 2   ;# host ubuntu-huy 14 GB RAM: -jobs 8 da lam treo may (2026-10-14)
set root [file normalize .]
create_project g3_ov $root/vivado_$tag -part xczu7ev-ffvc1156-2-e -force
set bp [lindex [lsort [get_board_parts -quiet *zcu104*]] end]
puts "board_part = $bp"
set_property board_part $bp [current_project]
set_property ip_repo_paths [lmap T $Ts {set x $root/prj_${pfx}T$T/sol/impl/ip}] [current_project]
update_ip_catalog
create_bd_design g3
create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == zynq_ultra_ps_e}] ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells ps]
set_property -dict [list CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $mhz] [get_bd_cells ps]
puts "PS intf: [get_bd_intf_pins ps/*]"
foreach T $Ts {
  create_bd_cell -type ip -vlnv [get_ipdefs -filter "NAME == g3_decode_${pfx}T$T"] g3_T$T
  apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list Clk_master {/ps/pl_clk0} Clk_slave {/ps/pl_clk0} Clk_xbar {/ps/pl_clk0} \
     Master {/ps/M_AXI_HPM0_FPD} Slave /g3_T$T/s_axi_control ddr_seg {Auto} intc_ip {Auto} master_apm {0}] [get_bd_intf_pins g3_T$T/s_axi_control]
}
# duong du lieu: noi tay qua SmartConnect 2 SI -> 1 MI -> HP0
create_bd_cell -type ip -vlnv [get_ipdefs -filter {NAME == smartconnect}] sc_mem
set_property -dict [list CONFIG.NUM_SI [llength $Ts] CONFIG.NUM_MI {1}] [get_bd_cells sc_mem]
set i 0
foreach T $Ts { connect_bd_intf_net [get_bd_intf_pins g3_T$T/m_axi_gmem] [get_bd_intf_pins sc_mem/S0${i}_AXI]; incr i }
connect_bd_intf_net [get_bd_intf_pins sc_mem/M00_AXI] [get_bd_intf_pins ps/S_AXI_HP0_FPD]
connect_bd_net [get_bd_pins ps/pl_clk0] [get_bd_pins sc_mem/aclk] [get_bd_pins ps/saxihp0_fpd_aclk]
set rst [lindex [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}] 0]
connect_bd_net [get_bd_pins $rst/peripheral_aresetn] [get_bd_pins sc_mem/aresetn]
assign_bd_address
validate_bd_design
save_bd_design
set wrap [make_wrapper -files [get_files g3.bd] -top]
add_files -norecurse $wrap
# synth OOC tung IP (mac dinh), chay lan luot -jobs 1: tong hop gop 1 luot can > 8 GB (OOM 2026-10-14)
set_property top g3_wrapper [current_fileset]
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
puts "IMPL_STATUS = [get_property STATUS [get_runs impl_1]]"
open_run impl_1
file mkdir $root/overlay_$tag
report_timing_summary -file $root/overlay_$tag/timing.rpt
report_utilization -file $root/overlay_$tag/util.rpt
puts "WNS = [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]"
file copy -force $root/vivado_$tag/g3_ov.runs/impl_1/g3_wrapper.bit $root/overlay_$tag/$tag.bit
file copy -force [glob $root/vivado_$tag/g3_ov.gen/sources_1/bd/g3/hw_handoff/g3.hwh] $root/overlay_$tag/$tag.hwh
puts "OVERLAY_DONE"
