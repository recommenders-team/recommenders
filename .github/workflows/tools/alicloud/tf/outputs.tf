# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Outputs the VM IP
#####################################################################
output "ssh_dest" {
  value = (
    alicloud_instance.reco != null
    ? "root@${alicloud_instance.reco.public_ip}"
    : ""
  )
  sensitive = true
}

output "ssh_key" {
  value = local.key_file_name
}

output "prices" {
  value = [
    for i, v in local.qualified_instance_types: {
      instance_type = v.id,
      instance_type_family = v.family,
      region = var.region
      price = data.alicloud_cloud_control_prices.reco[i].prices[0].trade_price
    }
  ]
  sensitive = true
}
