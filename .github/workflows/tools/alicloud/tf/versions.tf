# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Terraform providers
#####################################################################
terraform {
  required_providers {
    alicloud = {
      # https://registry.terraform.io/providers/aliyun/alicloud/latest
      source = "aliyun/alicloud"
      version = "2.0.0-beta5"
    }

    random = {
      source  = "hashicorp/random"
      version = ">=3.9.1"
    }
  }
}

provider "alicloud" {
  # See https://help.aliyun.com/en/document_detail/40654.html
  region = var.region
}
