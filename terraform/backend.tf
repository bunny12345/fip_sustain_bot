terraform {
  backend "s3" {
    bucket         = "chatvista-terraform-state"
    key            = "terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "chatvista-terraform-lock"
  }
}