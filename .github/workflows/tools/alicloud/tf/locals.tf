# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
#####################################################################
# Local variables used in the configuration
#####################################################################
# Assume that
# * The resource group, VPCs, VSwitches, and security groups are
#   created in advance.
# * All the VPCs, VSwitches, and security groups are managed in the
#   resource group.
# * VPCs are named after the regions where they are.
#------------------------------------------------------------------
locals {
  image_id = data.alicloud_images.ubuntu2404.ids[0]

  key_file_name = var.vm_name
  key_pair_name = var.vm_name

  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
  resource_group_name = "reco"

  security_group_ids = data.alicloud_security_groups.reco.ids

  vpc_id = data.alicloud_vpcs.reco.ids[0]
  vpc_name = var.region

  vswitch_id = data.alicloud_vswitches.reco.ids[0]
}

#------------------------------------------------------------------
# 
#------------------------------------------------------------------
locals {
  instance_type = [
    for it in data.alicloud_instance_types.reco.instance_types: it.id
    if it.cpu_core_count > 4 && it.memory_size >= 30 && it.price < 20
  ][0]
  instance_zone_id = local.instance_type.availability_zones[0]
}
