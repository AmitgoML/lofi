# L1 raw-immutable pipeline storage (S3)

> Scaffold — apply after ADR-001 (L2 store) and bucket naming confirmed with Lofi infra.

variable "pipeline_raw_bucket_name" {
  description = "S3 bucket for L1 raw-immutable platform API responses"
  type        = string
  default     = ""
}

variable "pipeline_raw_enabled" {
  description = "Create pipeline raw bucket resources"
  type        = bool
  default     = false
}

# Uncomment and configure when pipeline_raw_enabled = true:
#
# resource "aws_s3_bucket" "pipeline_raw" {
#   count  = var.pipeline_raw_enabled ? 1 : 0
#   bucket = var.pipeline_raw_bucket_name
#   tags   = var.tags
# }
#
# resource "aws_s3_bucket_versioning" "pipeline_raw" {
#   count  = var.pipeline_raw_enabled ? 1 : 0
#   bucket = aws_s3_bucket.pipeline_raw[0].id
#   versioning_configuration { status = "Enabled" }
# }
#
# resource "aws_s3_bucket_lifecycle_configuration" "pipeline_raw" {
#   count  = var.pipeline_raw_enabled ? 1 : 0
#   bucket = aws_s3_bucket.pipeline_raw[0].id
#   rule {
#     id     = "expire-noncurrent-versions"
#     status = "Enabled"
#     noncurrent_version_expiration { noncurrent_days = 90 }
#   }
# }
