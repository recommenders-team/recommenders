# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

terraform {
  required_providers {
    alicloud = {
      # https://registry.terraform.io/providers/aliyun/alicloud/latest
      source = "aliyun/alicloud"
      version = ">= 1.286.0"
    }
  }
}

provider "alicloud" {
  # See https://help.aliyun.com/en/document_detail/40654.html
  region = var.region
}
