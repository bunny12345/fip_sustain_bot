data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_s3_bucket" "faiss" {
  bucket = var.s3_bucket_faiss
  acl    = "private"

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "aws_s3_bucket_public_access_block" "faiss_block" {
  bucket                  = aws_s3_bucket.faiss.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

resource "aws_ecr_repository" "chatbot_lambda" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = false
  }
}

resource "aws_iam_role" "lambda_execution" {
  name               = "${var.project}-lambda-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy" "lambda_basic_execution" {
  name = "${var.project}-lambda-basic-execution"
  role = aws_iam_role.lambda_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_s3_access" {
  name = "${var.project}-lambda-s3-access"
  role = aws_iam_role.lambda_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.s3_bucket_faiss}",
          "arn:aws:s3:::${var.s3_bucket_faiss}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_bedrock_access" {
  name = "${var.project}-lambda-bedrock-access"
  role = aws_iam_role.lambda_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_marketplace_access" {
  name = "${var.project}-lambda-marketplace-access"
  role = aws_iam_role.lambda_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "aws-marketplace:ViewSubscriptions",
          "aws-marketplace:Subscribe"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_lambda_function" "chatbot" {
  function_name = var.lambda_function_name
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.chatbot_lambda.repository_url}:${var.lambda_image_tag}"
  role          = aws_iam_role.lambda_execution.arn
  timeout       = 120
  memory_size   = 3000
  architectures = ["x86_64"]

  environment {
    variables = {
      S3_BUCKET      = var.s3_bucket_faiss
      S3_KEY         = var.s3_key_faiss
      EMBED_MODEL_ID = var.embed_model_id
      LLM_MODEL_ID   = var.llm_model_id
    }
  }
}

resource "aws_apigatewayv2_api" "chatbot_http_api" {
  name          = var.api_name
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["https://learninggateway.eu"]   # restrict to your origin
    allow_methods = ["POST", "OPTIONS"]               # include POST and OPTIONS
    allow_headers = ["Content-Type", "Authorization"] # headers your client sends
    expose_headers = ["Content-Type"]
    max_age = 3600
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id                 = aws_apigatewayv2_api.chatbot_http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.chatbot.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "post_chat" {
  api_id    = aws_apigatewayv2_api.chatbot_http_api.id
  route_key = "POST /chat"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.chatbot_http_api.id
  name        = var.api_stage_name
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chatbot.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.chatbot_http_api.execution_arn}/*/*"
}
