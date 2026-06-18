data "aws_caller_identity" "current" {}

# Role that running containers use to call AWS APIs (Secrets Manager)
resource "aws_iam_role" "apprunner_instance" {
  name = "apprunner-instance-${var.app_name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{
      Effect    = "Allow",
      Principal = { Service = "tasks.apprunner.amazonaws.com" },
      Action    = "sts:AssumeRole"
    }]
  })
}

# Allow reading the secret (+ decrypt if CMK)
resource "aws_iam_role_policy" "apprunner_secret_access" {
  name = "apprunner-secret-access-${var.app_name}"
  role = aws_iam_role.apprunner_instance.id

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid      = "ReadAllSecretsInAccountRegion",
        Effect   = "Allow",
        Action   = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
          "secretsmanager:ListSecrets"
        ],
        Resource = "arn:aws:secretsmanager:${var.region}:${data.aws_caller_identity.current.account_id}:secret:*"
      }
    ]
  })
}