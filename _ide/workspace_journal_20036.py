# 2025-11-14T09:32:02.877585800
import vitis

client = vitis.create_client()
client.set_workspace(path="ScalarDudley")

status = client.add_platform_repos(platform=["c:\Xilinx\2025.1\Vitis\platforms"])

status = client.add_platform_repos(platform=["c:\Xilinx\2025.1\Vitis\platforms"])

status = client.add_platform_repos(platform=["c:\Users\Usuario\Desktop\kria-vitis-platforms\kv260\platforms"])

status = client.add_platform_repos(platform=["c:\Users\Usuario\Desktop\kria-vitis-platforms\kv260\platforms\kv260_bist"])

status = client.add_platform_repos(platform=["c:\Users\Usuario\Desktop\kria-vitis-platforms"])

advanced_options = client.create_advanced_options_dict(dt_overlay="0")

platform = client.create_platform_component(name = "platform_dmm",hw_design = "zed",os = "standalone",cpu = "ps7_cortexa9_0",domain_name = "standalone_ps7_cortexa9_0",generate_dtb = False,advanced_options = advanced_options,compiler = "gcc")

proj = client.create_sys_project(name="system_project", platform="$COMPONENT_LOCATION/../platform_dmm/export/platform_dmm/platform_dmm.xpfm", template="empty_accelerated_application" , build_output_type="xsa")

vitis.dispose()

