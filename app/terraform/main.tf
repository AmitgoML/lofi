terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.30"
    }
  }
  backend "s3" {
    bucket       = "lofi-terraform-config"
    key          = "dev/terraform.tfstate"
    region       = "us-east-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.region
}

# ---------- ECR ----------
resource "aws_ecr_repository" "this" {
  name                 = var.app_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# (Optional) lifecycle policy to keep last N images
resource "aws_ecr_lifecycle_policy" "keep_recent" {
  repository = aws_ecr_repository.this.name
  policy     = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 30 images"
      selection = {
        tagStatus     = "any"
        countType     = "imageCountMoreThan"
        countNumber   = 30
      }
      action = {
        type = "expire"
      }
    }]
  })
}

# ---------- IAM role that lets App Runner pull from ECR ----------
data "aws_iam_policy" "apprunner_ecr_policy" {
  arn = "arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"
}

resource "aws_iam_role" "apprunner_ecr_role" {
  name = "AppRunnerECRAccess-${var.app_name}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect = "Allow",
      Principal = { Service = "build.apprunner.amazonaws.com" },
      Action   = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "attach_policy" {
  role       = aws_iam_role.apprunner_ecr_role.name
  policy_arn = data.aws_iam_policy.apprunner_ecr_policy.arn
}

# ---------- (Optional) VPC connector for private DBs ----------
resource "aws_apprunner_vpc_connector" "this" {
  count = var.vpc_connector_enabled ? 1 : 0

  vpc_connector_name = "${var.app_name}-vpc"
  subnets            = var.vpc_subnet_ids
  security_groups    = var.vpc_security_group_ids
}

# ---------- App Runner service ----------
# NOTE: App Runner needs an existing image tag in ECR. Use create_service=false on first apply,
# push an image, then set create_service=true and re-apply.

locals {
  image_identifier = "${aws_ecr_repository.this.repository_url}:${var.image_tag}"
  env_vars_list = [
    for k, v in var.env_vars : {
      Name  = k
      Value = v
    }
  ]
}

resource "aws_apprunner_service" "this" {
  count       = var.create_service ? 1 : 0
  service_name = var.app_name

  source_configuration {
    image_repository {
      image_repository_type = "ECR"
      image_identifier      = local.image_identifier
      image_configuration {
        port = var.container_port
        runtime_environment_variables = local.env_vars_list
      }
    }
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_ecr_role.arn
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu    = var.cpu     # "1 vCPU" | "2 vCPU"
    memory = var.memory  # "2 GB" | "4 GB"
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = var.health_check_path
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  dynamic "network_configuration" {
    for_each = var.vpc_connector_enabled ? [1] : []
    content {
      egress_configuration {
        egress_type       = "VPC"
        vpc_connector_arn = aws_apprunner_vpc_connector.this[0].vpc_connector_arn
      }
    }
  }

  tags = var.tags
}
