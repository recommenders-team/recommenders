# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Create the following resources for the VM:
# *  VPC to form a private network with the following associated
#   resources:
#   + a VSwitch in the corresponding availability zone.
#   + a security group and a security group rule to allow SSH access.
#####################################################################
resource "random_integer" "vswitch_netnum" {
  # https://registry.terraform.io/providers/hashicorp/random/latest/docs/resources/integer
  min = 1
  max = pow(2, local.vpc_subnet_bits) - 1
}

resource "alicloud_vpc" "reco" {
  # VPC is regional.
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/vpc

  # Use the specified VPC if specified, or a new one will be created.
  count = var.vpc_id == "" ? 1 : 0

  vpc_name = var.unique_name
  cidr_block = "10.0.0.0/8"
}

resource "alicloud_vswitch" "reco" {
  # VSwitch is regional.
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/vswitch
  vswitch_name = local.vswitch_name
  vpc_id = local.vpc.id
  zone_id = local.instance_zone_id
  cidr_block = local.vswitch_cidr_block
}

resource "alicloud_security_group" "reco" {
  # Security groups seem to be regional, since they are associated
  # with VPCs.
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/security_group
  security_group_name = local.security_group_name
  vpc_id = local.vpc.id
}

resource "alicloud_security_group_rule" "ssh" {
  # Security group rules seem to be regional, since they are
  # associated with security groups.
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/security_group_rule
  security_group_id = local.security_group_id
  type = "ingress"
  ip_protocol = "tcp"
  policy = "accept"
  port_range = "22/22"
  cidr_ip = "0.0.0.0/0"
}


#------------------------------------------------------------------
# Create the VM and its SSH key
#------------------------------------------------------------------
resource "alicloud_ecs_key_pair" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair
  key_pair_name = local.key_pair_name
  key_file = local.key_file_name

  provisioner "local-exec" {
    when = destroy
    command = "rm -f ${self.key_file}"
  }
}

resource "alicloud_ecs_key_pair_attachment" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/ecs_key_pair_attachment
  key_pair_name = local.key_pair_name
  force = true
  instance_ids  = [alicloud_instance.reco.id]
}

resource "alicloud_instance" "reco" {
  # https://registry.terraform.io/providers/aliyun/alicloud/latest/docs/resources/instance
  instance_name = local.vm_name
  instance_type = local.instance_type_id
  image_id = local.image_id

  instance_charge_type = local.instance_charge_type
  spot_strategy = local.spot_strategy
  spot_duration = 1

  internet_charge_type = "PayByTraffic"
  internet_max_bandwidth_out = 5

  vpc_id = local.vpc.id
  vswitch_id = local.vswitch_id
  security_groups = [ local.security_group_id ]

  system_disk_category = "cloud_essd"
  system_disk_performance_level = "PL1"
  system_disk_size = 100
}
