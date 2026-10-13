# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Data queries for existing resources
#####################################################################
data "alicloud_vpcs" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/vpcs
  vpc_name = var.vpc_id == null ? var.unique_name : null
  ids = var.vpc_id == null ? null : [ var.vpc_id ]

  depends_on = [ alicloud_vpc.reco ]
}

data "alicloud_images" "ubuntu2404" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/images
  # https://ecs.console.aliyun.com/imageCatalog/region/cn-hangzhou?categoryType=system&osCategory%5B0%5D=linux&architecture%5B0%5D=x86_64&keyword=ubuntu
  # For example, "ubuntu_24_04_x64_20G_alibase_20260615.vhd",
  # note that the image names end with a date 20260615,
  # so we need to use this data source to get the latest one.
  name_regex = "^ubuntu_24_04_x64.*with_gpu_driver"
  os_type = "linux"
  architecture = "x86_64"
  most_recent = true
  owners = "system"
}

data "alicloud_instance_types" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/instance_types
  instance_type = var.instance_type
  instance_type_family = var.instance_type_family
  sorted_by = "Price"
  instance_charge_type = local.instance_charge_type
  spot_strategy = local.spot_strategy
  image_id = local.image_id
}

data "alicloud_cloud_control_prices" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/cloud_control_prices
  count = length(local.qualified_instance_types)

  desire_attributes_json = jsonencode({
    instance_type = local.qualified_instance_types[count.index].id
    instance_charge_type = local.instance_charge_type
    spot_strategy = local.spot_strategy
    system_disk = {
      category = "cloud_essd"
    }
  })
  product = "ECS"
  resource_code = "Instance"
}