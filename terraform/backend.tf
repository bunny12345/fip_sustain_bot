terraform {
  backend "s3" {
    bucket         = "fipbot-terraform-state"
    key            = "terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "fipbot-terraform-state-lock"
  }
}