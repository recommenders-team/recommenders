# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Outputs the VM IP
#####################################################################
output "ssh_dest" {
  value = "root@${alicloud_instance.reco.public_ip}"
}

output "ssh_key" {
  value = local.key_file_name
}
