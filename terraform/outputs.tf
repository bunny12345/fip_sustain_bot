output "lambda_function_name" {
  description = "The deployed Lambda function name."
  value       = aws_lambda_function.chatbot.function_name
}

output "ecr_repository_uri" {
  description = "The ECR repository URI for the Lambda image."
  value       = aws_ecr_repository.chatbot_lambda.repository_url
}

output "api_endpoint" {
  description = "The HTTP API endpoint for the chatbot."
  value       = format("%s/%s", aws_apigatewayv2_api.chatbot_http_api.api_endpoint, aws_apigatewayv2_stage.prod.name)
}

output "lambda_role_arn" {
  description = "The Lambda execution role ARN."
  value       = aws_iam_role.lambda_execution.arn
}
