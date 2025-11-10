# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

COPY requirements.txt .

# Install dependencies if requirements.txt exists
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Expose port 
EXPOSE 8000
WORKDIR /app/RandomForestClassifier
RUN python RandomForestClassifier.py

WORKDIR /app/LogisticRegression
RUN python LogisticRegression.py

WORKDIR /app/GradientBoostingClassifier
RUN python GradientBoostingClassifier.py

WORKDIR /app
CMD ["python", "test.py"]