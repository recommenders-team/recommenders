# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
#####################################################################
# Creat the VM and its SSH key
#####################################################################
resource "alicloud_ecs_key_pair" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair
  key_pair_name = local.key_pair_name
  key_file = local.key_file_name
  resource_group_id = local.resource_group_id
}

resource "alicloud_ecs_key_pair_attachment" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair_attachment
  key_pair_name = local.key_pair_name
  force = true
  instance_ids  = [alicloud_instance.reco.id]
}

resource "alicloud_instance" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/instance
  instance_name = var.vm_name
  resource_group_id = local.resource_group_id
  instance_type = local.instance_type
  image_id = local.image_id

  instance_charge_type = "PostPaid"
  spot_strategy = "SpotAsPriceGo"
  spot_duration = 1

  internet_charge_type = "PayByTraffic"
  internet_max_bandwidth_out = 5

  vpc_id = local.vpc_id
  vswitch_id = local.vswitch_id
  security_groups = local.security_group_ids

  system_disk_category = "cloud_essd"
  system_disk_performance_level = "PL1"
  system_disk_size = 100
}
