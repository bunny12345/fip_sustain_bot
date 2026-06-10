variable "aws_region" {
  type    = string
  default = "eu-west-1"
}

variable "project" {
  type    = string
  default = "chatvista-ai"
}

variable "s3_bucket_faiss" {
  type    = string
  default = "faissindexingirlcollege"
}

variable "s3_bucket_docs" {
  type    = string
  default = "irlcolleges"
}

variable "s3_key_faiss" {
  type    = string
  default = "faiss_index.tar.gz"
}

variable "ecr_repo_name" {
  type    = string
  default = "chatbot-lambda"
}

variable "lambda_function_name" {
  type    = string
  default = "chatbot-lambda"
}

variable "lambda_image_tag" {
  type    = string
  default = "latest"
}

variable "api_name" {
  type    = string
  default = "chatbot-http-api"
}

variable "api_stage_name" {
  type    = string
  default = "prod"
}

variable "embed_model_id" {
  type    = string
  default = "cohere.embed-v4:0"
}

variable "llm_model_id" {
  type    = string
  default = "anthropic.claude-3-sonnet-20240229-v1:0"
}
