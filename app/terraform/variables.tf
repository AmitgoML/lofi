variable "region" {
  type        = string
  description = "AWS region"
  default     = "us-east-1"
}

variable "app_name" {
  type        = string
  description = "App Runner service / ECR repo name"
  default     = "lucy"
}

# Deploy control
variable "create_service" {
  type        = bool
  description = "Whether to create the App Runner service now (needs an existing image tag in ECR)"
  default     = false
}

variable "image_tag" {
  type        = string
  description = "ECR image tag to deploy (must exist)"
  default     = "latest"
}

# Container settings
variable "container_port" {
  type        = string
  description = "Container port your app listens on"
  default     = "8000"
}

variable "cpu" {
  type        = string
  description = "Instance vCPU for App Runner: e.g., '1 vCPU' or '2 vCPU'"
  default     = "1 vCPU"
}

variable "memory" {
  type        = string
  description = "Instance memory for App Runner: e.g., '2 GB' or '4 GB'"
  default     = "2 GB"
}

variable "health_check_path" {
  type        = string
  description = "HTTP health check path"
  default     = "/health"
}

variable "env_vars" {
  type        = map(string)
  description = "Environment variables for the container"
  default     = {
    ENV       = "prod"
    LOG_LEVEL = "info"
    PORT      = "8000"
  }
}

# Optional VPC connector (for RDS/Redis access)
variable "vpc_connector_enabled" {
  type        = bool
  description = "Attach a VPC connector for private egress"
  default     = false
}

variable "vpc_subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for the VPC connector"
  default     = []
}

variable "vpc_security_group_ids" {
  type        = list(string)
  description = "Security group IDs used by the VPC connector"
  default     = []
}

variable "tags" {
  type        = map(string)
  description = "Tags to apply to resources"
  default     = {}
}
