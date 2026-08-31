# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

data "alicloud_resource_manager_resource_groups" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/resource_manager_resource_groups
  name_regex = var.resource_group_name
}

data "alicloud_vpcs" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/vpcs
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
  vpc_name = var.region
}

data "alicloud_vswitches" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/vswitches
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
  vswitch_name = var.region
}

data "alicloud_security_groups" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/security_groups
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
  name_regex = var.region
}

data "alicloud_images" "ubuntu2404" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/images
  # https://ecs.console.aliyun.com/imageCatalog/region/cn-hangzhou?categoryType=system&osCategory%5B0%5D=linux&architecture%5B0%5D=x86_64&keyword=ubuntu
  # For example, "ubuntu_24_04_x64_20G_alibase_20260615.vhd",
  # note that the image names end with a date 20260615,
  # so we need to use this data source to get the latest one.
  name_regex = "^ubuntu_24_04.*20G"
  os_type = "linux"
  architecture = "x86_64"
  most_recent = true
  owners = "system"
}

data "alicloud_instance_types" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/data-sources/instance_types
  instance_type_family = var.instance_type_family
  sorted_by = "Price"
  instance_charge_type = "PostPaid"
  spot_strategy = "SpotAsPriceGo"
  image_id = data.alicloud_images.ubuntu2404.ids[0]
}
