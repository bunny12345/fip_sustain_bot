FROM public.ecr.aws/lambda/python:3.11

# Copy function code
COPY haiku_model.py .
COPY lambda_function.py .
COPY requirements.txt .

# Install build dependencies required for compiling scipy, numpy, and other packages
RUN yum install -y gcc gcc-c++ make python3-devel lapack-devel blas-devel libgomp

# Install Python packages using binary wheels where available
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --only-binary=:all: numpy==1.26.4 faiss-cpu==1.7.4 && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

# Download and install AWS Lambda RIE for local testing
RUN curl -Lo /usr/bin/aws-lambda-rie https://github.com/aws/aws-lambda-runtime-interface-emulator/releases/latest/download/aws-lambda-rie && \
    chmod +x /usr/bin/aws-lambda-rie

# Set the Lambda handler (filename.function_name)
CMD ["lambda_function.lambda_handler"]