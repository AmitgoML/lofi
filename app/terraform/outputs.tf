output "ecr_repository_url" {
  value       = aws_ecr_repository.this.repository_url
  description = "ECR repo URL (push here)"
}

output "apprunner_service_url" {
  value       = try(aws_apprunner_service.this[0].service_url, null)
  description = "Public HTTPS URL of the App Runner service"
}

output "apprunner_service_arn" {
  value       = try(aws_apprunner_service.this[0].arn, null)
  description = "App Runner service ARN"
}
