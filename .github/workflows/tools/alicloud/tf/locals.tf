# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Local variables used in the configuration
#####################################################################
# VM related variables
#------------------------------------------------------------------
locals {
  vm_name = var.unique_name
  key_file_name = var.unique_name
  key_pair_name = var.unique_name

  instance_charge_type = "PostPaid"
  spot_strategy = "SpotAsPriceGo"

  image_id = (
    length(data.alicloud_images.ubuntu2404.ids) != 0
    ? data.alicloud_images.ubuntu2404.ids[0]
    : null
  )
}


#------------------------------------------------------------------
# Requirements that the VM should satisfy
#------------------------------------------------------------------
locals {
  qualified_instance_types = [
    for it in data.alicloud_instance_types.reco.instance_types: it
      if it.cpu_core_count > 4
      && it.memory_size >= 30
      && it.price < 22
      && local.image_id != null
  ]

  instance_type = (
    length(local.qualified_instance_types) != 0
    ? local.qualified_instance_types[0]
    : null
  )
  instance_type_id = (
    local.instance_type != null
    ? local.instance_type.id
    : null
  )
  instance_zone_id = (
    local.instance_type != null
    ? local.instance_type.availability_zones[0]
    : null
  )
}


#------------------------------------------------------------------
# Network related variables
# * If a VPC is provided, use the specified VPC because the VPC is
#   the network where the mirrors are.
#------------------------------------------------------------------
locals {
  vpc = (
    length(data.alicloud_vpcs.reco.vpcs) != 0
    ? data.alicloud_vpcs.reco.vpcs[0]
    : null
  )
  vpc_subnet_bits = (
    local.vpc != null
    ? 29 - convert(split("/", local.vpc.cidr_block)[1], number)
    : null
  )

  vswitch_name = var.unique_name
  vswitch_cidr_block = (
    local.vpc != null
    ? cidrsubnet(
      local.vpc.cidr_block,
      local.vpc_subnet_bits,
      random_integer.vswitch_netnum
    )
    : null
  )
  vswitch_id = (
    length(data.alicloud_vswitches.reco.ids) != 0
    ? data.alicloud_vswitches.reco.ids[0]
    : null
  )

  security_group_name = var.unique_name
  security_group_id = (
    length(data.alicloud_security_groups.reco.ids) != 0
    ? data.alicloud_security_groups.reco.ids[0]
    : null
  )
}
