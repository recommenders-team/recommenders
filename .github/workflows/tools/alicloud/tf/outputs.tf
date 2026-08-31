# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

output "ip" {
  value = alicloud_instance.reco.public_ip
}
