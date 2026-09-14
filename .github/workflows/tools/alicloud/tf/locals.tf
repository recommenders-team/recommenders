# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Local variables used in the configuration
#####################################################################
# * All resources except the VPC are managed in the resource group.
# * If a VPC is provided, use the specified VPC because the VPC is
#   the network where the mirrors are.
#------------------------------------------------------------------
locals {
  image_id = data.alicloud_images.ubuntu2404.ids[0]
  instance_charge_type = "PostPaid"
  spot_strategy = "SpotAsPriceGo"

  key_file_name = var.unique_name
  key_pair_name = var.unique_name

  resource_group_name = var.unique_name
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]

  security_group_name = var.unique_name
  security_group_id = data.alicloud_security_groups.reco.ids[0]


  vm_name = var.unique_name

  vpc = data.alicloud_vpcs.reco.vpcs[0]
  vpc_name = var.vpc_id == "" ? var.unique_name : local.vpc.vpc_name
  vpc_id = local.vpc.id
  vpc_cidr_block = var.vpc_id == "" ? "172.16.0.0/4" : local.vpc.cidr_block
  vpc_mask_length = split(local.vpc_cidr_block, "/")[1]
  vpc_subnet_bits = 30 - convert(local.vpc_mask_length, number)

  vswitch_name = var.unique_name
  vswitch_cidr_block = (
    var.vswitch_id == "" ?
    "172.16.0.0/30" : cidrsubnet(
      local.vpc_cidr_block,
      local.vpc_subnet_bits,
      random_integer.vswitch_netnum
    )
  )
  vswitch_id = data.alicloud_vswitches.reco.ids[0]
}

#------------------------------------------------------------------
# Requirements that the VM should satisfy
#------------------------------------------------------------------
locals {
  instance_type = [
    for it in data.alicloud_instance_types.reco.instance_types: it.id
    if it.cpu_core_count > 4 && it.memory_size >= 30 && it.price < 20
  ][0]
  instance_zone_id = local.instance_type.availability_zones[0]
}
