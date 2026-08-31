# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

resource "alicloud_ecs_key_pair" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair
  key_pair_name = var.vm_name
  key_file = var.vm_name
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
}

resource "alicloud_ecs_key_pair_attachment" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair_attachment
  key_pair_name = var.vm_name
  force = true
  instance_ids  = [alicloud_instance.reco.id]
}

resource "alicloud_instance" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/instance
  instance_name = var.vm_name
  resource_group_id = data.alicloud_resource_manager_resource_groups.reco.ids[0]
  instance_type = [for it in data.alicloud_instance_types.reco.instance_types: it.id if it.memory_size >= 30 && it.price < 20][0]
  image_id = data.alicloud_images.ubuntu2404.ids[0]

  instance_charge_type = "PostPaid"
  spot_strategy = "SpotAsPriceGo"
  spot_duration = 1

  internet_charge_type = "PayByTraffic"
  internet_max_bandwidth_out = 5

  vpc_id = data.alicloud_vpcs.reco.ids[0]
  vswitch_id = data.alicloud_vswitches.reco.ids[0]
  security_groups = data.alicloud_security_groups.reco.ids

  system_disk_category = "cloud_essd"
  system_disk_performance_level = "PL1"
  system_disk_size = 100
}
