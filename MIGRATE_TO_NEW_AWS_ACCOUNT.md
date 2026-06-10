# Migrate ChatVista AI to a New AWS Account

This document explains the exact code and configuration changes needed to move this setup to another AWS account.
It includes all the required AWS names, Terraform backend, GitHub secrets, and code updates.

## 1. Use fresh, unique names

The following values must be unique when moving to a new account:
- `terraform/backend.tf` S3 bucket: `chatvista-terraform-state`
- `terraform/variables.tf` FAISS bucket: `faissindexingirlcollege`
- `terraform/variables.tf` documents bucket: `irlcolleges`
- `lambda_function.py` / `main.py` S3 bucket names if they are still hard-coded

For example, in the new account use names like:
- `chatvista-terraform-state-<your-suffix>`
- `chatvista-terraform-lock-<your-suffix>`
- `chatvista-faiss-index-<your-suffix>`
- `chatvista-docs-<your-suffix>`

> S3 bucket names are global, so they must be different from any existing bucket name in AWS.

## 2. Create the backend state bucket and lock table in the new account

Run these commands in the new AWS account first:

```bash
aws s3 mb s3://chatvista-terraform-state-<your-suffix> --region eu-west-1
aws dynamodb create-table \
  --table-name chatvista-terraform-lock-<your-suffix> \
  --attribute-definitions AttributeName=LockID,AttributeType=S \
  --key-schema AttributeName=LockID,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region eu-west-1
```

## 3. Update `terraform/backend.tf`

Replace the bucket and DynamoDB table with the new account names.

Example:

```hcl
terraform {
  backend "s3" {
    bucket         = "chatvista-terraform-state-<your-suffix>"
    key            = "terraform.tfstate"
    region         = "eu-west-1"
    dynamodb_table = "chatvista-terraform-lock-<your-suffix>"
  }
}
```

## 4. Update `terraform/variables.tf`

Change these values to new names for the target account:

```hcl
variable "aws_region" {
  default = "eu-west-1"
}

variable "project" {
  default = "chatvista-ai"
}

variable "s3_bucket_faiss" {
  default = "chatvista-faiss-index-<your-suffix>"
}

variable "s3_bucket_docs" {
  default = "chatvista-docs-<your-suffix>"
}

variable "s3_key_faiss" {
  default = "faiss_index.tar.gz"
}

variable "ecr_repo_name" {
  default = "chatbot-lambda"
}

variable "lambda_function_name" {
  default = "chatbot-lambda"
}

variable "api_name" {
  default = "chatbot-http-api"
}
```

> You can keep `ecr_repo_name`, `lambda_function_name`, and `api_name` the same if the new AWS account is empty. The only globally unique names required are S3 bucket names.

## 5. Update code references to the new account/buckets

### `main.py`

Change the S3 document bucket and key used to build the FAISS index:

```python
loader = S3FileLoader(
    bucket="chatvista-docs-<your-suffix>",
    key="<your-document-key>.pdf"
)
```

### `lambda_function.py`

Update the FAISS bucket and key:

```python
S3_BUCKET = "chatvista-faiss-index-<your-suffix>"
S3_KEY = "faiss_index.tar.gz"
AWS_REGION = "eu-west-1"
```

If you also use `haiku_model.py`, `llama_model_1.py`, or `llama_model_2.py`, update the `S3_BUCKET` values there too.

## 6. Update GitHub Actions secrets for the new account

Set these repository secrets in GitHub for the new AWS account:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION` (e.g. `eu-west-1`)
- `FAISS_S3_BUCKET` (used by `00_full_deploy.yml`)

If you have a separate repo for the new account, make sure these secrets are configured there.

## 7. Run Terraform init in the new account

From the repo root:

```bash
cd terraform
rm -rf .terraform .terraform.lock.hcl
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  hashicorp/terraform:1.7.0 init
```

When prompted, confirm state migration if you are moving existing local state to the new backend.

## 8. Deploy to the new account

After updating the config, run:

```bash
cd terraform
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  hashicorp/terraform:1.7.0 plan -out=tfplan

docker run --rm -v "$PWD":/workspace -w /workspace \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  hashicorp/terraform:1.7.0 apply -auto-approve tfplan
```

Then run the GitHub workflow once the secrets are updated.

## 9. Checklist before pushing

- [ ] `terraform/backend.tf` points to a new S3 backend bucket
- [ ] `terraform/variables.tf` uses unique bucket names
- [ ] `main.py` references the new documents bucket/key
- [ ] `lambda_function.py` references the new FAISS bucket/key
- [ ] GitHub secrets exist for the new AWS account
- [ ] `aws s3 ls s3://chatvista-terraform-state-<your-suffix>/` shows the state file after `terraform init`

## 10. If the new account already has manual resources

If you already created some resources manually in the new account, import them instead of recreating them.

Example:

```bash
docker run --rm -v "$PWD":/workspace -w /workspace \
  -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_REGION \
  hashicorp/terraform:1.7.0 import aws_ecr_repository.chatbot_lambda chatbot-lambda
```

Do this for any existing AWS resources that you want Terraform to manage.

---

If you want, I can also update your repo to parameterize the hard-coded S3 bucket names in `lambda_function.py` and `main.py` so migration is easier next time.